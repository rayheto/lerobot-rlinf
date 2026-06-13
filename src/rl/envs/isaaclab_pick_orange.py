"""SO-101 PickOrange RL env wrapper — sparse multi-orange reward (v4).

Reward events (all sparse, no per-step dense shaping):

  +grasp_bonus       per orange, first frame that orange is grasped
  +carry_speed_coef * |Δee|   each step while ANY orange grasped
  +place_bonus       per orange, first frame that orange lands on plate
  -drop_penalty      per orange, was_grasped & !is_grasped & !placed
  +rest_bonus        on first frame all 3 placed AND arm at rest pose
  -timeout_penalty   if episode truncates without all 3 placed + rest
  -fail_penalty      fail-A: any orange falls below tabletop or out of workspace

Term: rest_emitted (success) | fail_emitted (fail-A) | step >= max_episode_steps.

AuxObs side-channel publishes orange001/2/3 + plate + EE pose + joints so the
parent process can compute reward / shaping without reading ``env.scene[...]``
across IPC.

Official task ref: third_party/leisaac/.../mdp/terminations.py:task_done
(3 oranges within (±0.10, ±0.10, [-0.07, 0.07]) of plate + so101 at rest).
"""
from __future__ import annotations

import gymnasium as gym
import torch

from rlinf.envs.isaaclab.isaaclab_env import IsaaclabBaseEnv


# ---------------------------------------------------------------------------
# Unit conversion: leisaac (radians, URDF joint range) ↔ lerobot (motor degrees).
# pi05 SFT was trained on lerobot motor-degree convention; the raw env runs in
# leisaac radians. Without these conversions, pi05 sees garbage state (rad
# treated as degrees) and the env sees garbage action (degrees treated as rad,
# ~57× scaled-down). We wrap conversions at the env boundary so downstream
# code (policy, BC anchor, OOD KD-tree) all see consistent lerobot units.
# ---------------------------------------------------------------------------


# Hardcoded mirror of leisaac.assets.robots.lerobot.SO101_FOLLOWER_USD_JOINT_LIMLITS
# and SO101_FOLLOWER_MOTOR_LIMITS. Hardcoded because the upstream module imports
# isaaclab.sim at the top-level, which pulls in omni — only available inside the
# Isaac App subprocess. These values are device-table constants.
_SO101_USD_JOINT_LIMS_DEG = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10.0, 100.0),
}
_SO101_MOTOR_LIMS = {
    "shoulder_pan": (-100.0, 100.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 100.0),
    "wrist_flex": (-100.0, 100.0),
    "wrist_roll": (-100.0, 100.0),
    "gripper": (0.0, 100.0),
}


def _lazy_so101_limits():
    """Build per-joint limit tensors (no isaaclab imports)."""
    joints = list(_SO101_USD_JOINT_LIMS_DEG.keys())
    joint_lo = torch.tensor([_SO101_USD_JOINT_LIMS_DEG[j][0] for j in joints], dtype=torch.float32)
    joint_hi = torch.tensor([_SO101_USD_JOINT_LIMS_DEG[j][1] for j in joints], dtype=torch.float32)
    motor_lo = torch.tensor([_SO101_MOTOR_LIMS[j][0] for j in joints], dtype=torch.float32)
    motor_hi = torch.tensor([_SO101_MOTOR_LIMS[j][1] for j in joints], dtype=torch.float32)
    return joint_lo, joint_hi, motor_lo, motor_hi


def _leisaac_rad_to_lerobot_deg(x: torch.Tensor, lims) -> torch.Tensor:
    """(..., 6) radians → lerobot motor units. Joint limits are in degrees, so
    convert rad→deg first then linearly rescale per joint into motor range.
    Mirrors leisaac's `convert_leisaac_action_to_lerobot`."""
    joint_lo, joint_hi, motor_lo, motor_hi = lims
    jlo = joint_lo.to(x.device); jhi = joint_hi.to(x.device)
    mlo = motor_lo.to(x.device); mhi = motor_hi.to(x.device)
    x_deg = x * (180.0 / torch.pi)
    joint_range = jhi - jlo
    motor_range = mhi - mlo
    return (x_deg - jlo) / joint_range * motor_range + mlo


def _lerobot_deg_to_leisaac_rad(x: torch.Tensor, lims) -> torch.Tensor:
    """(..., 6) motor degrees → radians (URDF joint range). Mirrors
    `convert_lerobot_action_to_leisaac`."""
    joint_lo, joint_hi, motor_lo, motor_hi = lims
    jlo = joint_lo.to(x.device); jhi = joint_hi.to(x.device)
    mlo = motor_lo.to(x.device); mhi = motor_hi.to(x.device)
    joint_range = jhi - jlo
    motor_range = mhi - mlo
    motor_in_range = x - mlo
    deg_in_range = motor_in_range / motor_range * joint_range + jlo
    return deg_in_range * (torch.pi / 180.0)


# ---------------------------------------------------------------------------
# AuxObs term functions (top-level for pickling).
# ---------------------------------------------------------------------------


def _root_pos_w(env, asset_cfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


def _ee_pos_w(env, ee_frame_cfg) -> torch.Tensor:
    ee_frame = env.scene[ee_frame_cfg.name]
    # target index 1 = "jaw" with detection offset (SingleArmTaskSceneCfg).
    return ee_frame.data.target_pos_w[:, 1, :] - env.scene.env_origins


def _gripper_pos(env, robot_cfg) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    return robot.data.joint_pos[:, -1:].clone()


def _joint_pos_full(env, robot_cfg) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    return robot.data.joint_pos.clone()


def _build_aux_obs_group():
    from isaaclab.managers import (
        ObservationGroupCfg as ObsGroup,
        ObservationTermCfg as ObsTerm,
        SceneEntityCfg,
    )
    from isaaclab.utils import configclass

    @configclass
    class _AuxObsGroup(ObsGroup):
        orange001_pos = ObsTerm(func=_root_pos_w, params={"asset_cfg": SceneEntityCfg("Orange001")})
        orange002_pos = ObsTerm(func=_root_pos_w, params={"asset_cfg": SceneEntityCfg("Orange002")})
        orange003_pos = ObsTerm(func=_root_pos_w, params={"asset_cfg": SceneEntityCfg("Orange003")})
        plate_pos = ObsTerm(func=_root_pos_w, params={"asset_cfg": SceneEntityCfg("Plate")})
        ee_pos = ObsTerm(func=_ee_pos_w, params={"ee_frame_cfg": SceneEntityCfg("ee_frame")})
        gripper_pos = ObsTerm(func=_gripper_pos, params={"robot_cfg": SceneEntityCfg("robot")})
        joint_pos_full = ObsTerm(func=_joint_pos_full, params={"robot_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    return _AuxObsGroup()


# Rest pose ranges in lerobot motor-degree convention (joint order matches
# _SO101_USD_JOINT_LIMS_DEG / aux["joint_pos_full"]).
# Mirrors leisaac SO101_FOLLOWER_REST_POSE_RANGE (rad-based) but evaluated
# directly in URDF degrees after rad→deg conversion, since aux joint_pos_full
# is already in URDF radians.
_REST_POSE_DEG = {
    "shoulder_pan":  (-30.0, 30.0),
    "shoulder_lift": (-130.0, -70.0),
    "elbow_flex":    (60.0, 120.0),
    "wrist_flex":    (20.0, 80.0),
    "wrist_roll":    (-30.0, 30.0),
    "gripper":       (-40.0, 20.0),
}


def _arm_at_rest(joint_pos_rad: torch.Tensor) -> torch.Tensor:
    """(Nenv, 6) URDF radians → (Nenv,) bool — all 6 joints within rest range."""
    joints = list(_SO101_USD_JOINT_LIMS_DEG.keys())
    deg = joint_pos_rad * (180.0 / torch.pi)
    ok = torch.ones(deg.shape[0], dtype=torch.bool, device=deg.device)
    for i, jname in enumerate(joints):
        lo, hi = _REST_POSE_DEG[jname]
        ok &= (deg[:, i] > lo) & (deg[:, i] < hi)
    return ok


# ---------------------------------------------------------------------------
# RL env
# ---------------------------------------------------------------------------


class IsaaclabPickOrangeEnv(IsaaclabBaseEnv):
    """Sparse 3-stage reward wrapper for SO-101 PickOrange."""

    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        rcfg = cfg.get("reward", {})
        self._grasp_bonus = float(rcfg.get("grasp_bonus", 10.0))
        self._carry_speed_coef = float(rcfg.get("carry_speed_coef", 0.5))
        self._place_bonus = float(rcfg.get("place_bonus", 20.0))
        self._drop_penalty = float(rcfg.get("drop_penalty", -5.0))
        self._timeout_penalty = float(rcfg.get("timeout_penalty", -2.0))
        self._rest_bonus = float(rcfg.get("rest_bonus", 30.0))
        self._fail_penalty = float(rcfg.get("fail_penalty", -5.0))

        # Grasp predicate thresholds — kept from v2 (mirror leisaac.mdp.orange_grasped).
        self._grasp_diff_threshold = float(rcfg.get("grasp_diff_threshold", 0.05))
        self._grasp_close_threshold = float(rcfg.get("grasp_close_threshold", 0.60))
        self._grasp_lift_threshold = float(rcfg.get("grasp_lift_threshold", 0.06))

        # Place predicate (mirrors leisaac put_orange_to_plate).
        self._place_xy = float(rcfg.get("place_xy", 0.10))
        self._place_z_lo = float(rcfg.get("place_z_lo", -0.07))
        self._place_z_hi = float(rcfg.get("place_z_hi", 0.07))

        # Fail-A workspace bounds, in env-local meters (asset.root_pos_w - env.origin).
        # z floor: any orange dropping >5cm below its init-z = off the table.
        # xy box: ±0.6 m around env origin (counter top is comfortably inside).
        self._fail_z_drop = float(rcfg.get("fail_z_drop", 0.05))
        self._fail_xy = float(rcfg.get("fail_xy", 0.40))  # max XY displacement from init

        super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)

        Nenv = self.num_envs
        dev = self.device
        # Per-env × per-orange (3) flags
        self._grasp_emitted = torch.zeros(Nenv, 3, dtype=torch.bool, device=dev)
        self._place_emitted = torch.zeros(Nenv, 3, dtype=torch.bool, device=dev)
        self._was_grasped   = torch.zeros(Nenv, 3, dtype=torch.bool, device=dev)
        # Per-env terminal flags
        self._rest_emitted  = torch.zeros(Nenv, dtype=torch.bool, device=dev)
        self._fail_emitted  = torch.zeros(Nenv, dtype=torch.bool, device=dev)
        self._orange_init_z: torch.Tensor | None = None   # (Nenv, 3)
        self._orange_init_xy: torch.Tensor | None = None  # (Nenv, 3, 2)
        self._prev_ee_pos: torch.Tensor | None = None

        self._unit_lims = _lazy_so101_limits()

    # ------------------------------------------------------------------
    # Subprocess env construction
    # ------------------------------------------------------------------

    def _make_env_function(self):
        cfg_init = self.cfg.init_params
        task_id = self.isaaclab_env_id

        def make_env_isaaclab():
            import os

            os.environ.pop("DISPLAY", None)

            from isaaclab.app import AppLauncher

            sim_app = AppLauncher(headless=True, enable_cameras=True).app
            from isaaclab_tasks.utils import load_cfg_from_registry
            import leisaac  # noqa: F401  registers LeIsaac-* gym ids

            isaac_env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
            isaac_env_cfg.seed = self.seed
            isaac_env_cfg.scene.num_envs = cfg_init.num_envs
            if "episode_length_s" in cfg_init:
                isaac_env_cfg.episode_length_s = float(cfg_init.episode_length_s)
            # decimation controls how many physics steps per env.step. leisaac
            # SO101 task default is 1 (60Hz outer), but SFT data is 30 fps —
            # set 2 so each env.step advances 1/30s sim, matching pi05 chunk
            # tempo. See docs/dryrun_jax_step8f_crossanalysis.md §6.
            if "decimation" in cfg_init:
                isaac_env_cfg.decimation = int(cfg_init.decimation)

            if "wrist_cam" in cfg_init:
                isaac_env_cfg.scene.wrist.width = cfg_init.wrist_cam.width
                isaac_env_cfg.scene.wrist.height = cfg_init.wrist_cam.height
            if "front_cam" in cfg_init:
                isaac_env_cfg.scene.front.width = cfg_init.front_cam.width
                isaac_env_cfg.scene.front.height = cfg_init.front_cam.height

            isaac_env_cfg.observations.aux = _build_aux_obs_group()

            from leisaac.devices.action_process import init_action_cfg
            isaac_env_cfg.actions = init_action_cfg(isaac_env_cfg.actions, "so101leader")

            env = gym.make(task_id, cfg=isaac_env_cfg, render_mode="rgb_array").unwrapped
            return env, sim_app

        return make_env_isaaclab

    # ------------------------------------------------------------------
    # Obs wrapping
    # ------------------------------------------------------------------

    def _wrap_obs(self, obs):
        self._last_aux = obs.get("aux", {})

        policy = obs["policy"]
        instruction = [self.task_description] * self.num_envs

        # Convert joint state to the lerobot motor-degree convention pi05 was
        # trained on. _last_aux["joint_pos_full"] is in leisaac (URDF) radians.
        states_deg = _leisaac_rad_to_lerobot_deg(
            self._last_aux["joint_pos_full"], self._unit_lims
        )

        return {
            "main_images": policy["front"],
            "task_descriptions": instruction,
            "states": states_deg,
            "wrist_images": policy["wrist"],
        }

    # ------------------------------------------------------------------
    # Step / reset
    # ------------------------------------------------------------------

    def step(self, actions=None, auto_reset=True):
        # Convert policy action (lerobot motor degrees) → leisaac radians for the
        # underlying env. None / zero-action passes through unchanged.
        if actions is not None:
            actions = _lerobot_deg_to_leisaac_rad(actions, self._unit_lims)
        raw_obs, raw_reward, raw_term, raw_trunc, _infos = self.env.step(actions)
        del raw_reward  # leisaac has no RewardsCfg; ignore.

        obs = self._wrap_obs(raw_obs)

        self._elapsed_steps += 1
        truncations = (self.elapsed_steps >= self.cfg.max_episode_steps) | raw_trunc

        step_reward, terminal_mask = self._compute_step_reward(truncations)
        success_mask = self._rest_emitted.clone()
        fail_mask = self._fail_emitted.clone()
        terminations = terminal_mask if not self.ignore_terminations else torch.zeros_like(terminal_mask)

        dones = terminations | truncations

        infos = self._record_metrics(step_reward, terminations, {})
        # Override the base class's "success = any positive reward" rule:
        # success only when rest_emitted (all 3 placed + arm at rest).
        self.success_once = self._rest_emitted.clone()
        infos["episode"]["success_once"] = self._rest_emitted.clone()
        # `fail_once` is the broad "this episode ended in failure" signal — any
        # terminal step that wasn't a success. This includes (a) fail-A (orange
        # off-table / out of xy box) AND (b) timeout-without-success. Keep the
        # narrow `failA_once` for breakdown.
        infos["episode"]["fail_once"] = dones & (~self._rest_emitted)
        infos["episode"]["failA_once"] = self._fail_emitted.clone()
        infos["episode"]["success_at_end"] = success_mask
        infos["episode"]["fail_at_end"] = fail_mask
        if self.ignore_terminations:
            terminations = torch.zeros_like(terminal_mask)

        if dones.any() and auto_reset and self.auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)

        return obs, step_reward, terminations, truncations, infos

    def _stack_orange_pos(self):
        a = self._last_aux
        return torch.stack([a["orange001_pos"], a["orange002_pos"], a["orange003_pos"]], dim=1)  # (Nenv,3,3)

    def reset(self, seed=None, env_ids=None):
        obs, infos = super().reset(seed=seed, env_ids=env_ids)
        ee_pos = self._last_aux["ee_pos"]
        orange_xyz = self._stack_orange_pos()  # (Nenv,3,3)
        orange_z = orange_xyz[..., 2]          # (Nenv,3)
        orange_xy = orange_xyz[..., :2]        # (Nenv,3,2)
        if env_ids is None:
            self._grasp_emitted[:] = False
            self._place_emitted[:] = False
            self._was_grasped[:] = False
            self._rest_emitted[:] = False
            self._fail_emitted[:] = False
            self._orange_init_z = orange_z.clone()
            self._orange_init_xy = orange_xy.clone()
            self._prev_ee_pos = ee_pos.clone()
        else:
            self._grasp_emitted[env_ids] = False
            self._place_emitted[env_ids] = False
            self._was_grasped[env_ids] = False
            self._rest_emitted[env_ids] = False
            self._fail_emitted[env_ids] = False
            if self._orange_init_z is None:
                self._orange_init_z = orange_z.clone()
            else:
                self._orange_init_z[env_ids] = orange_z[env_ids]
            if self._orange_init_xy is None:
                self._orange_init_xy = orange_xy.clone()
            else:
                self._orange_init_xy[env_ids] = orange_xy[env_ids]
            if self._prev_ee_pos is None:
                self._prev_ee_pos = ee_pos.clone()
            else:
                self._prev_ee_pos[env_ids] = ee_pos[env_ids]
        return obs, infos

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_step_reward(self, truncations: torch.Tensor):
        aux = self._last_aux
        orange_pos = self._stack_orange_pos()   # (Nenv,3,3) — env-local meters
        plate_pos = aux["plate_pos"]            # (Nenv,3)
        ee_pos = aux["ee_pos"]                  # (Nenv,3)
        gripper = aux["gripper_pos"][:, 0]      # (Nenv,)
        joint_pos = aux["joint_pos_full"]       # (Nenv,6) URDF rad

        if self._orange_init_z is None:
            self._orange_init_z = orange_pos[..., 2].clone()
        if self._orange_init_xy is None:
            self._orange_init_xy = orange_pos[..., :2].clone()
        if self._prev_ee_pos is None:
            self._prev_ee_pos = ee_pos.clone()

        # --- per-orange predicates ((Nenv, 3) bool tensors) ---
        ee_b = ee_pos.unsqueeze(1)              # (Nenv,1,3) broadcast
        d_ee_o = torch.linalg.vector_norm(ee_b - orange_pos, dim=-1)  # (Nenv,3)
        lifted = (orange_pos[..., 2] - self._orange_init_z) > self._grasp_lift_threshold
        gripper_closed = (gripper < self._grasp_close_threshold).unsqueeze(1)  # (Nenv,1)
        is_grasped = (d_ee_o < self._grasp_diff_threshold) & gripper_closed & lifted

        plate_b = plate_pos.unsqueeze(1)         # (Nenv,1,3)
        d_xy = torch.linalg.vector_norm(orange_pos[..., :2] - plate_b[..., :2], dim=-1)  # (Nenv,3)
        rel_z = orange_pos[..., 2] - plate_b[..., 2]
        on_plate = (
            (d_xy <= self._place_xy)
            & (rel_z >= self._place_z_lo)
            & (rel_z <= self._place_z_hi)
        )

        # --- 1. grasp bonus (first frame per orange) ---
        grasp_now = is_grasped & (~self._grasp_emitted)
        r_grasp = self._grasp_bonus * grasp_now.float().sum(dim=1)  # (Nenv,)
        self._grasp_emitted = self._grasp_emitted | is_grasped

        # --- 2. carry speed reward while ANY orange grasped ---
        ee_disp = torch.linalg.vector_norm(ee_pos - self._prev_ee_pos, dim=-1)  # (Nenv,)
        any_grasped = is_grasped.any(dim=1).float()
        r_carry = self._carry_speed_coef * ee_disp * any_grasped
        self._prev_ee_pos = ee_pos.clone()

        # --- 3. place bonus (first frame per orange, requires prior grasp) ---
        place_now = on_plate & self._grasp_emitted & (~self._place_emitted)
        r_place = self._place_bonus * place_now.float().sum(dim=1)
        self._place_emitted = self._place_emitted | place_now

        # --- 4. drop penalty per orange (was grasped, no longer, not placed) ---
        dropped = self._was_grasped & (~is_grasped) & (~self._place_emitted)
        r_drop = self._drop_penalty * dropped.float().sum(dim=1)
        self._was_grasped = is_grasped

        # --- 5. rest bonus: all 3 placed AND arm at rest (first frame) ---
        all_placed = self._place_emitted.all(dim=1)
        at_rest = _arm_at_rest(joint_pos)
        rest_now = all_placed & at_rest & (~self._rest_emitted)
        r_rest = self._rest_bonus * rest_now.float()
        self._rest_emitted = self._rest_emitted | rest_now

        # --- 6. fail-A: any orange off-table or moved out of workspace ---
        # z: below init by more than fail_z_drop -> fell off the table
        # xy: moved away from init position by more than fail_xy radius (relative,
        # NOT env-local absolute — kitchen scene puts oranges far from env origin)
        z_drop = (orange_pos[..., 2] - self._orange_init_z) < (-self._fail_z_drop)
        xy_shift = torch.linalg.vector_norm(
            orange_pos[..., :2] - self._orange_init_xy, dim=-1
        )  # (Nenv, 3)
        xy_out = xy_shift > self._fail_xy
        fail_now_per_o = (z_drop | xy_out) & (~self._place_emitted)
        fail_now = fail_now_per_o.any(dim=1) & (~self._fail_emitted) & (~self._rest_emitted)
        r_fail = self._fail_penalty * fail_now.float()
        self._fail_emitted = self._fail_emitted | fail_now

        # --- 7. timeout penalty (truncated, not yet finished/failed) ---
        timed_out = truncations & (~self._rest_emitted) & (~self._fail_emitted)
        r_timeout = self._timeout_penalty * timed_out.float()

        total = r_grasp + r_carry + r_place + r_drop + r_rest + r_fail + r_timeout
        # success_mask = task done (rest bonus emitted at least once)
        # terminal_mask = success OR fail-A (both end the episode)
        terminal_mask = self._rest_emitted | self._fail_emitted
        return total, terminal_mask
