"""SO-101 PickOrange RL env wrapper — sparse 3-stage reward (v3).

Reward events (all sparse, no per-step dense shaping):

  +grasp_bonus       on first frame ``is_grasped`` flips true
  +carry_speed_coef * |Δee|   each step while grasped
  +place_bonus       on first frame the orange is on the plate
  -drop_penalty      when was_grasped & !is_grasped (re-dropped)
  -timeout_penalty   if episode truncates without ever placing

Term: place_emitted (success) | step >= max_episode_steps (truncation).
KL anchoring to the SFT reference policy is handled by RLinf's PPO loss
(``algorithm.kl_beta``) — this env is reward-only.

AuxObs side-channel publishes orange/plate/EE pose + joints so the parent
process can compute reward without reading ``env.scene[...]`` across IPC.
"""
from __future__ import annotations

import gymnasium as gym
import torch

from rlinf.envs.isaaclab.isaaclab_env import IsaaclabBaseEnv


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
        plate_pos = ObsTerm(func=_root_pos_w, params={"asset_cfg": SceneEntityCfg("Plate")})
        ee_pos = ObsTerm(func=_ee_pos_w, params={"ee_frame_cfg": SceneEntityCfg("ee_frame")})
        gripper_pos = ObsTerm(func=_gripper_pos, params={"robot_cfg": SceneEntityCfg("robot")})
        joint_pos_full = ObsTerm(func=_joint_pos_full, params={"robot_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    return _AuxObsGroup()


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

        # Grasp predicate thresholds — kept from v2 (mirror leisaac.mdp.orange_grasped).
        self._grasp_diff_threshold = float(rcfg.get("grasp_diff_threshold", 0.05))
        self._grasp_close_threshold = float(rcfg.get("grasp_close_threshold", 0.60))
        self._grasp_lift_threshold = float(rcfg.get("grasp_lift_threshold", 0.06))

        super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)

        self._grasp_emitted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._place_emitted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._was_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._orange_init_z: torch.Tensor | None = None
        self._prev_ee_pos: torch.Tensor | None = None

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

        return {
            "main_images": policy["front"],
            "task_descriptions": instruction,
            "states": self._last_aux["joint_pos_full"],
            "wrist_images": policy["wrist"],
        }

    # ------------------------------------------------------------------
    # Step / reset
    # ------------------------------------------------------------------

    def step(self, actions=None, auto_reset=True):
        raw_obs, raw_reward, raw_term, raw_trunc, _infos = self.env.step(actions)
        del raw_reward  # leisaac has no RewardsCfg; ignore.

        obs = self._wrap_obs(raw_obs)

        self._elapsed_steps += 1
        truncations = (self.elapsed_steps >= self.cfg.max_episode_steps) | raw_trunc

        step_reward, success_mask = self._compute_step_reward(truncations)
        terminations = success_mask if not self.ignore_terminations else torch.zeros_like(success_mask)

        dones = terminations | truncations

        infos = self._record_metrics(step_reward, terminations, {})
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = success_mask
            terminations = torch.zeros_like(success_mask)

        if dones.any() and auto_reset and self.auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)

        return obs, step_reward, terminations, truncations, infos

    def reset(self, seed=None, env_ids=None):
        obs, infos = super().reset(seed=seed, env_ids=env_ids)
        ee_pos = self._last_aux["ee_pos"]
        orange_z = self._last_aux["orange001_pos"][:, 2]
        if env_ids is None:
            self._grasp_emitted[:] = False
            self._place_emitted[:] = False
            self._was_grasped[:] = False
            self._orange_init_z = orange_z.clone()
            self._prev_ee_pos = ee_pos.clone()
        else:
            self._grasp_emitted[env_ids] = False
            self._place_emitted[env_ids] = False
            self._was_grasped[env_ids] = False
            if self._orange_init_z is None:
                self._orange_init_z = orange_z.clone()
            else:
                self._orange_init_z[env_ids] = orange_z[env_ids]
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
        orange_pos = aux["orange001_pos"]
        plate_pos = aux["plate_pos"]
        ee_pos = aux["ee_pos"]
        gripper = aux["gripper_pos"][:, 0]

        if self._orange_init_z is None:
            self._orange_init_z = orange_pos[:, 2].clone()
        if self._prev_ee_pos is None:
            self._prev_ee_pos = ee_pos.clone()

        # --- predicates ---
        d_ee_orange = torch.linalg.vector_norm(ee_pos - orange_pos, dim=-1)
        lifted = (orange_pos[:, 2] - self._orange_init_z) > self._grasp_lift_threshold
        is_grasped = (
            (d_ee_orange < self._grasp_diff_threshold)
            & (gripper < self._grasp_close_threshold)
            & lifted
        )

        d_xy = torch.linalg.vector_norm(orange_pos[:, :2] - plate_pos[:, :2], dim=-1)
        rel_z = orange_pos[:, 2] - plate_pos[:, 2]
        on_plate = (d_xy <= 0.10) & (rel_z >= -0.07) & (rel_z <= 0.25)

        # --- 1. grasp bonus (first frame) ---
        grasp_now = is_grasped & (~self._grasp_emitted)
        r_grasp = self._grasp_bonus * grasp_now.float()
        self._grasp_emitted = self._grasp_emitted | is_grasped

        # --- 2. carry speed reward while grasped ---
        ee_disp = torch.linalg.vector_norm(ee_pos - self._prev_ee_pos, dim=-1)
        r_carry = self._carry_speed_coef * ee_disp * is_grasped.float()
        self._prev_ee_pos = ee_pos.clone()

        # --- 3. place bonus (first frame) ---
        place_now = on_plate & self._grasp_emitted & (~self._place_emitted)
        r_place = self._place_bonus * place_now.float()
        self._place_emitted = self._place_emitted | place_now

        # --- 4. drop penalty (was grasped, no longer) ---
        dropped = self._was_grasped & (~is_grasped) & (~self._place_emitted)
        r_drop = self._drop_penalty * dropped.float()
        self._was_grasped = is_grasped

        # --- 5. timeout penalty ---
        timed_out = truncations & (~self._place_emitted)
        r_timeout = self._timeout_penalty * timed_out.float()

        total = r_grasp + r_carry + r_place + r_drop + r_timeout
        success_mask = self._place_emitted.clone()
        return total, success_mask
