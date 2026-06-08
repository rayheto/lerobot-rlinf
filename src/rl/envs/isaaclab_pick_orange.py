"""SO-101 PickOrange RL env wrapper.

Wraps the leisaac ``LeIsaac-SO101-PickOrange-v0`` IsaacLab task with:

- A staged reward (reach / grasp / lift / align / place / rest / success / OOD
  / step) that turns the diagnostic findings in
  ``docs/sft_diagnostics_findings.md`` into RL signals. The KNN OOD penalty
  reuses the same intra-reference 1-NN sigma as EXP_05, so the reward and the
  diagnostic are numerically aligned.
- An optional single-orange mode (``cfg.single_orange: True``) that overrides
  the leisaac termination — leisaac's ``task_done`` requires all three oranges
  on the plate plus return-to-rest, which is too sparse for Phase 1 PPO.
- A non-invasive AuxObs group injected at sub-process env construction time:
  the wrapper extends ``observations`` with a new ``aux`` group that publishes
  orange/plate root poses, EE pose, and rest-pose flag. No leisaac code is
  modified — the new group is appended in our ``_make_env_function``.

IPC note: the IsaacLab env runs in a SubProcIsaacLabEnv subprocess. We cannot
read ``env.scene[...]`` from the wrapper process. Everything we need for
reward computation must travel back through the obs dict — that's why the
AuxObsCfg exists.
"""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import torch

# NOTE: do NOT import any `isaaclab.*` or `leisaac.*` at module top-level —
# isaaclab.assets/envs/managers all transitively pull in ``omni`` which is
# only available after ``AppLauncher`` boots the Kit app in the subprocess.
# All such imports live inside ``make_env_isaaclab`` below. Pattern mirrors
# third_party/RLinf/rlinf/envs/isaaclab/tasks/stack_cube.py.
from rlinf.envs.isaaclab.isaaclab_env import IsaaclabBaseEnv

from src.rl.envs.ood_kdtree import OodKNNConfig, OodKNNPenalty


# ---------------------------------------------------------------------------
# AuxObs term functions (top-level for pickling into the subprocess).
# These access env.scene at *call time* (after sim_app boots) so they need no
# isaaclab imports here — duck-typed on the asset_cfg/robot_cfg `.name` attr.
# ---------------------------------------------------------------------------


def _root_pos_w(env, asset_cfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


def _rest_pose_flag(env, robot_cfg) -> torch.Tensor:
    from leisaac.utils.robot_utils import is_so101_at_rest_pose

    robot = env.scene[robot_cfg.name]
    flag = is_so101_at_rest_pose(robot.data.joint_pos, robot.data.joint_names)
    return flag.float().unsqueeze(-1)


def _ee_pos_w(env, ee_frame_cfg) -> torch.Tensor:
    ee_frame = env.scene[ee_frame_cfg.name]
    # target index 1 = "jaw" with detection offset (see SingleArmTaskSceneCfg).
    return ee_frame.data.target_pos_w[:, 1, :] - env.scene.env_origins


def _gripper_pos(env, robot_cfg) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    return robot.data.joint_pos[:, -1:].clone()


def _joint_pos_full(env, robot_cfg) -> torch.Tensor:
    # 6-dim SO-101 joint pos in URDF order — matches SFT observation.state.
    robot = env.scene[robot_cfg.name]
    return robot.data.joint_pos.clone()


def _build_aux_obs_group():
    """Construct the AuxObs configclass at subprocess runtime (post-AppLauncher).

    Done inside a function so the ``isaaclab.managers`` imports don't run at
    parent-process import time (they need ``omni``).
    """
    from isaaclab.managers import (
        ObservationGroupCfg as ObsGroup,
        ObservationTermCfg as ObsTerm,
        SceneEntityCfg,
    )
    from isaaclab.utils import configclass

    @configclass
    class _AuxObsGroup(ObsGroup):
        """Side-channel ObsGroup that publishes physical quantities to the wrapper."""

        orange001_pos = ObsTerm(func=_root_pos_w, params={"asset_cfg": SceneEntityCfg("Orange001")})
        orange002_pos = ObsTerm(func=_root_pos_w, params={"asset_cfg": SceneEntityCfg("Orange002")})
        orange003_pos = ObsTerm(func=_root_pos_w, params={"asset_cfg": SceneEntityCfg("Orange003")})
        plate_pos = ObsTerm(func=_root_pos_w, params={"asset_cfg": SceneEntityCfg("Plate")})
        ee_pos = ObsTerm(func=_ee_pos_w, params={"ee_frame_cfg": SceneEntityCfg("ee_frame")})
        gripper_pos = ObsTerm(func=_gripper_pos, params={"robot_cfg": SceneEntityCfg("robot")})
        joint_pos_full = ObsTerm(func=_joint_pos_full, params={"robot_cfg": SceneEntityCfg("robot")})
        rest_flag = ObsTerm(func=_rest_pose_flag, params={"robot_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    return _AuxObsGroup()


# ---------------------------------------------------------------------------
# RL env
# ---------------------------------------------------------------------------


class IsaaclabPickOrangeEnv(IsaaclabBaseEnv):
    """RL wrapper that injects staged reward + OOD penalty + optional single-orange mode."""

    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        # Reward / OOD config — read from cfg before super().__init__ so the
        # subprocess can pick them up at env construction time.
        rcfg = cfg.get("reward", {})
        self._reward_coefs = {
            "reach": float(rcfg.get("reach", -0.5)),
            "grasp": float(rcfg.get("grasp", 1.0)),
            "lift": float(rcfg.get("lift", 3.0)),
            "align": float(rcfg.get("align", -0.5)),
            "place": float(rcfg.get("place", 5.0)),
            "rest": float(rcfg.get("rest", 1.0)),
            "success": float(rcfg.get("success", 10.0)),
            "step": float(rcfg.get("step", -0.001)),
        }
        self._lift_h_max = float(rcfg.get("lift_h_max", 0.10))
        self._grasp_diff_threshold = float(rcfg.get("grasp_diff_threshold", 0.05))
        self._grasp_close_threshold = float(rcfg.get("grasp_close_threshold", 0.60))
        self._grasp_lift_threshold = float(rcfg.get("grasp_lift_threshold", 0.06))
        self._single_orange = bool(cfg.get("single_orange", True))

        ocfg = cfg.get("ood_reward", {})
        self._ood_cfg = (
            OodKNNConfig(
                demo_dataset_path=str(ocfg["demo_dataset_path"]),
                k_neighbors=int(ocfg.get("k_neighbors", 5)),
                coef=float(ocfg.get("coef", 1.0)),
            )
            if ocfg and ocfg.get("demo_dataset_path")
            else None
        )
        self._ood = OodKNNPenalty.get(self._ood_cfg) if self._ood_cfg is not None else None

        super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)

        # Per-env transition tracking (transitions are awarded once per episode).
        self._grasp_awarded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._place_awarded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._rest_awarded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._orange_init_z: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Env construction (runs in subprocess via cloudpickle)
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
            # Importing leisaac registers the LeIsaac-* gym ids (gym.register
            # calls live at import time in leisaac/tasks/.../__init__.py).
            import leisaac  # noqa: F401

            isaac_env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
            isaac_env_cfg.seed = self.seed
            isaac_env_cfg.scene.num_envs = cfg_init.num_envs

            # Camera sizing — match SFT training (224×224 after the OpenPI
            # transform); leisaac defaults are 640×480 which we keep at the
            # IsaacLab side and let the policy transform resize.
            if "wrist_cam" in cfg_init:
                isaac_env_cfg.scene.wrist.width = cfg_init.wrist_cam.width
                isaac_env_cfg.scene.wrist.height = cfg_init.wrist_cam.height
            if "front_cam" in cfg_init:
                isaac_env_cfg.scene.front.width = cfg_init.front_cam.width
                isaac_env_cfg.scene.front.height = cfg_init.front_cam.height

            # Inject AuxObs group — non-invasive extension.
            isaac_env_cfg.observations.aux = _build_aux_obs_group()

            # leisaac's SingleArmActionsCfg leaves arm_action / gripper_action
            # as MISSING; init_action_cfg(..., "so101leader") populates them
            # with the 5-joint arm + 1-joint gripper JointPositionActionCfg
            # that the SFT data was recorded against. We avoid calling
            # `use_teleop_device` because its side effects (disabling robot
            # gravity, swapping task_type) are inappropriate for policy rollout.
            from leisaac.devices.action_process import init_action_cfg
            isaac_env_cfg.actions = init_action_cfg(isaac_env_cfg.actions, "so101leader")

            env = gym.make(task_id, cfg=isaac_env_cfg, render_mode="rgb_array").unwrapped
            return env, sim_app

        return make_env_isaaclab

    # ------------------------------------------------------------------
    # Obs wrapping
    # ------------------------------------------------------------------

    def _wrap_obs(self, obs):
        # Stash raw aux for reward computation in step().
        self._last_aux = obs.get("aux", {})

        policy = obs["policy"]
        instruction = [self.task_description] * self.num_envs

        # SFT data path: front → base_0_rgb, wrist → left_wrist_0_rgb (matches
        # So101LiftInputs / dataconfig). We pass front as "main_images" and
        # wrist as "wrist_images" — the policy transform maps them on.
        front_image = policy["front"]
        wrist_image = policy["wrist"]
        joint_pos_6 = self._last_aux["joint_pos_full"]

        return {
            "main_images": front_image,
            "task_descriptions": instruction,
            "states": joint_pos_6,
            "wrist_images": wrist_image,
        }

    # ------------------------------------------------------------------
    # Reward + termination override
    # ------------------------------------------------------------------

    def step(self, actions=None, auto_reset=True):
        # We replace the base class step's reward and termination computation
        # but reuse its bookkeeping (elapsed_steps, metrics, auto_reset).
        raw_obs, raw_reward, raw_term, raw_trunc, _infos = self.env.step(actions)
        del raw_reward  # leisaac PickOrange has no RewardsCfg; raw_reward is zero.

        obs = self._wrap_obs(raw_obs)

        step_reward, success_mask = self._compute_step_reward()
        terminations = success_mask if not self.ignore_terminations else torch.zeros_like(success_mask)

        self._elapsed_steps += 1
        truncations = (self.elapsed_steps >= self.cfg.max_episode_steps) | raw_trunc

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
        # Reset transition latches and the initial-z anchor used by lift reward.
        if env_ids is None:
            self._grasp_awarded[:] = False
            self._place_awarded[:] = False
            self._rest_awarded[:] = False
            self._orange_init_z = self._last_aux["orange001_pos"][:, 2].clone()
        else:
            self._grasp_awarded[env_ids] = False
            self._place_awarded[env_ids] = False
            self._rest_awarded[env_ids] = False
            if self._orange_init_z is None:
                self._orange_init_z = self._last_aux["orange001_pos"][:, 2].clone()
            else:
                self._orange_init_z[env_ids] = self._last_aux["orange001_pos"][env_ids, 2]
        return obs, infos

    # ------------------------------------------------------------------
    # Reward internals
    # ------------------------------------------------------------------

    def _compute_step_reward(self):
        aux = self._last_aux
        c = self._reward_coefs

        orange_pos = aux["orange001_pos"]  # [N,3]
        plate_pos = aux["plate_pos"]
        ee_pos = aux["ee_pos"]
        gripper = aux["gripper_pos"][:, 0]  # [N]
        rest_flag = aux["rest_flag"][:, 0].bool()  # [N]
        joint_full = aux["joint_pos_full"]  # [N,6]

        if self._orange_init_z is None:
            self._orange_init_z = orange_pos[:, 2].clone()

        # r_reach
        d_ee_orange = torch.linalg.vector_norm(ee_pos - orange_pos, dim=-1)
        r_reach = c["reach"] * d_ee_orange

        # Grasp predicate (mirrors leisaac mdp.orange_grasped).
        lifted = (orange_pos[:, 2] - self._orange_init_z) > self._grasp_lift_threshold
        grasped_now = (d_ee_orange < self._grasp_diff_threshold) & (gripper < self._grasp_close_threshold) & lifted

        # r_grasp — transition reward (only at first frame grasp becomes true).
        r_grasp_mask = grasped_now & (~self._grasp_awarded)
        r_grasp = c["grasp"] * r_grasp_mask.float()
        self._grasp_awarded = self._grasp_awarded | grasped_now

        # r_lift — continuous bonus while grasped.
        lift_h = (orange_pos[:, 2] - self._orange_init_z).clamp(min=0.0, max=self._lift_h_max)
        r_lift = c["lift"] * lift_h * grasped_now.float()

        # r_align — only after first grasp.
        d_xy = torch.linalg.vector_norm(orange_pos[:, :2] - plate_pos[:, :2], dim=-1)
        r_align = c["align"] * d_xy * self._grasp_awarded.float()

        # r_place — orange on the plate. Mirrors mdp.put_orange_to_plate ranges.
        on_plate_xy = (d_xy <= 0.10)
        rel_z = orange_pos[:, 2] - plate_pos[:, 2]
        on_plate_z = (rel_z >= -0.07) & (rel_z <= 0.25)
        placed_now = on_plate_xy & on_plate_z
        r_place_mask = placed_now & (~self._place_awarded)
        r_place = c["place"] * r_place_mask.float()
        self._place_awarded = self._place_awarded | placed_now

        # r_rest — SO-101 returned to rest pose, only meaningful once we placed.
        rest_now = rest_flag & self._place_awarded
        r_rest_mask = rest_now & (~self._rest_awarded)
        r_rest = c["rest"] * r_rest_mask.float()
        self._rest_awarded = self._rest_awarded | rest_now

        # Success — Phase 1 (single_orange): placed + rest on Orange001 only.
        # Phase 2 would extend this to all three oranges.
        if self._single_orange:
            success_mask = self._place_awarded & self._rest_awarded
        else:
            ok2 = self._orange_in_plate(aux["orange002_pos"], plate_pos)
            ok3 = self._orange_in_plate(aux["orange003_pos"], plate_pos)
            success_mask = self._place_awarded & ok2 & ok3 & self._rest_awarded

        r_success = c["success"] * success_mask.float()

        # r_step — constant time penalty.
        r_step = torch.full_like(r_reach, c["step"])

        # r_ood
        if self._ood is not None:
            r_ood = self._ood.reward(joint_full)
        else:
            r_ood = torch.zeros_like(r_reach)

        total = r_reach + r_grasp + r_lift + r_align + r_place + r_rest + r_success + r_step + r_ood
        return total, success_mask

    @staticmethod
    def _orange_in_plate(orange_pos: torch.Tensor, plate_pos: torch.Tensor) -> torch.Tensor:
        d_xy = torch.linalg.vector_norm(orange_pos[:, :2] - plate_pos[:, :2], dim=-1)
        rel_z = orange_pos[:, 2] - plate_pos[:, 2]
        return (d_xy <= 0.10) & (rel_z >= -0.07) & (rel_z <= 0.25)
