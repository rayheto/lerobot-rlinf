"""Smoke test for LeIsaac's SO-101 lift_cube Isaac Lab env.

Reset, step 10 random actions, print obs schema, save first overhead
frame. Confirms `leisaac` package install + USD assets all resolved.

Run:
    /home/hlei/miniconda3/envs/rlinf-isaacsim-env/bin/python scripts/smoke_lift_cube_leisaac.py
"""
import os
import sys
import pathlib

sys.argv = sys.argv[:1]

# leisaac resolves ASSETS_ROOT via `git rev-parse --show-toplevel` → our repo
# root (= /home/hlei/robotic/lerobot-rlinf/assets) by default, but the scene
# USDs live under third_party/leisaac/assets/. Override before import.
_REPO = pathlib.Path(__file__).resolve().parents[1]
os.environ["LEISAAC_ASSETS_ROOT"] = str(_REPO / "third_party" / "leisaac" / "assets")

from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=True).app

import gymnasium as gym
import numpy as np
import cv2
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import leisaac  # noqa: F401 — triggers gym.register(LeIsaac-SO101-*)
from leisaac.devices.action_process import init_action_cfg

ENV_ID = "LeIsaac-SO101-LiftCube-v0"
OUT = pathlib.Path("/tmp/cam_compare")
OUT.mkdir(exist_ok=True)

print(f"[SMOKE] making env: {ENV_ID}")
env_cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1)
# `arm_action` / `gripper_action` are MISSING on the env cfg until a control
# scheme is chosen. so101leader → joint-position action on all 6 joints
# (matches LeRobot dataset action semantics).
init_action_cfg(env_cfg.actions, device="so101leader")
env = gym.make(ENV_ID, cfg=env_cfg).unwrapped
print(f"[SMOKE] action_space={env.action_space.shape}  obs keys (top): preparing reset")

obs, _ = env.reset()

def describe(x, indent=2):
    pad = " " * indent
    if isinstance(x, dict):
        for k, v in x.items():
            print(f"{pad}{k}:")
            describe(v, indent + 2)
    elif hasattr(x, "shape"):
        print(f"{pad}shape={tuple(x.shape)}  dtype={x.dtype}")
    else:
        print(f"{pad}{type(x).__name__} = {x!r}"[:120])

print("[SMOKE] obs schema after reset:")
describe(obs)

import torch  # imported here so AppLauncher boots Kit first
for i in range(10):
    a = torch.from_numpy(env.action_space.sample()).to("cuda:0")
    obs, r, term, trunc, _info = env.step(a)
    if i == 0:
        print(f"[SMOKE] step 0: reward={float(r):.3f}  term={bool(term)}  trunc={bool(trunc)}")

# Save the front camera frame. LeIsaac lift_cube exposes it as
# obs["policy"]["front"] with shape (B, H, W, 3) uint8.
policy = obs.get("policy", {})
for k, v in (policy.items() if isinstance(policy, dict) else []):
    if hasattr(v, "shape") and v.dim() == 4 and v.shape[-1] == 3 and v.dtype == torch.uint8:
        arr = v[0].cpu().numpy()
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        path = OUT / f"leisaac_lift_cube_{k}.png"
        cv2.imwrite(str(path), bgr)
        print(f"[SMOKE] saved {path}  shape={arr.shape}")

print("[SMOKE] OK")
env.close()
app.close()
