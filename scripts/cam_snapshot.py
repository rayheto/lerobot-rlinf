"""Boot sponge env, reset, save first overhead + wrist frames to PNG.

Run:
    /home/hlei/miniconda3/envs/rlinf-isaacsim-env/bin/python scripts/cam_snapshot.py
Outputs to /tmp/cam_compare/sim_{overhead,wrist}_frame0.png.
"""
import sys
import pathlib

sys.argv = sys.argv[:1]

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True, enable_cameras=True).app

import gymnasium as gym
import numpy as np
import cv2
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import lerobot_rlinf.tasks  # noqa: F401

OUT = pathlib.Path("/tmp/cam_compare")
OUT.mkdir(exist_ok=True)

env_cfg = parse_env_cfg("Isaac-Lift-Sponge-Bowl-SO101-Play-v0", device="cuda:0", num_envs=1)
env = gym.make("Isaac-Lift-Sponge-Bowl-SO101-Play-v0", cfg=env_cfg).unwrapped
obs, _ = env.reset()

for key in ("overhead", "wrist"):
    arr = obs["images"][key][0].cpu().numpy().astype(np.uint8)   # [H, W, 3] RGB
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    path = OUT / f"sim_{key}_frame0.png"
    cv2.imwrite(str(path), bgr)
    print(f"saved {path}  shape={arr.shape}  min={arr.min()}  max={arr.max()}")

env.close()
app.close()
