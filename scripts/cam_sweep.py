"""Sweep overhead camera poses and snapshot each, to find one that matches dataset.

Runs ONE Kit boot, mutates `/World/envs/env_0/overhead_cam` xform between
shots. Saves /tmp/cam_compare/sim_overhead_pose{N}.png + index.html.
"""
import sys
import pathlib
import math

sys.argv = sys.argv[:1]

from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=True).app

import gymnasium as gym
import numpy as np
import cv2
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import lerobot_rlinf.tasks  # noqa: F401

OUT = pathlib.Path("/tmp/cam_compare")
OUT.mkdir(exist_ok=True)

env_cfg = parse_env_cfg("Isaac-Lift-Sponge-Bowl-SO101-Play-v0", device="cuda:0", num_envs=1)
env = gym.make("Isaac-Lift-Sponge-Bowl-SO101-Play-v0", cfg=env_cfg).unwrapped


def quat_y_rot(theta_deg):
    """Quaternion (w, x, y, z) for rotation θ about world +Y axis."""
    h = math.radians(theta_deg / 2.0)
    return (math.cos(h), 0.0, math.sin(h), 0.0)


# Each candidate: (label, pos, quat-y-rotation-angle-degrees)
# rot is "rotate ROS cam +Z (forward) about world Y by θ".
# θ=90° → forward = world +X (looking down +X axis level). θ>90° tilts down.
# Want forward looking back toward robot (+X if camera in front of robot),
# OR forward looking forward (-X) if camera behind. Dataset suggests camera
# is BEHIND robot and far back, looking forward over the workspace.
CANDIDATES = [
    ("A_behind_far_lowtilt",  (-0.80,  0.00, 0.35), 100),  # behind, 10° down
    ("B_behind_far_midtilt",  (-0.80,  0.00, 0.40), 110),  # behind, 20° down
    ("C_behind_close_steep",  (-0.40,  0.00, 0.50), 130),  # original-ish
    ("D_left_behind",         (-0.60, -0.30, 0.40), 110),  # behind+left
    ("E_front_low",           ( 0.55,  0.00, 0.25), -100), # IN FRONT, looking back
    ("F_front_higher",        ( 0.65,  0.00, 0.35), -110),
]

# Override overhead_cam offset by directly setting its xform pose post-reset.
obs, _ = env.reset()
overhead_cam = env.scene["overhead_cam"]
print(f"overhead_cam type: {type(overhead_cam).__name__}")

for label, pos, theta in CANDIDATES:
    quat = quat_y_rot(theta)
    pos_t = torch.tensor([list(pos)], device="cuda:0", dtype=torch.float32)
    quat_t = torch.tensor([list(quat)], device="cuda:0", dtype=torch.float32)
    overhead_cam.set_world_poses(positions=pos_t, orientations=quat_t, convention="ros")
    # Step sim once so camera renders new view.
    env.sim.step()
    env.scene.update(0.0)
    img = overhead_cam.data.output["rgb"][0].cpu().numpy().astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    path = OUT / f"sim_overhead_{label}.png"
    cv2.imwrite(str(path), bgr)
    print(f"saved {path}  pos={pos}  theta={theta}°")

env.close()
app.close()
