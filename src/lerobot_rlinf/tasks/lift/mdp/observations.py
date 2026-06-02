# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_deg(
    env: ManagerBasedRLEnv,
    asset_name: str,
    joint_names: list[str],
) -> torch.Tensor:
    """Joint positions in degrees — LeRobot v3.0 SO-101 dataset convention.

    Output order = `joint_names` (canonical URDF order), independent of
    articulation's internal joint ordering. Matches `observation.state`
    captured by lerobot teleop.
    """
    import math
    asset: Articulation = env.scene[asset_name]
    name_to_id = {n: i for i, n in enumerate(asset.joint_names)}
    ids = [name_to_id[n] for n in joint_names]
    pos_rad = asset.data.joint_pos[:, ids]
    return pos_rad * (180.0 / math.pi)


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b
