"""Smoke test for any LeIsaac SO-101 Isaac Lab env.

Reset, step N random actions, print obs schema, save first front/wrist
camera frames. Confirms `leisaac` package install + USD assets resolved.

Run (lift_cube, headless):
    /home/hlei/miniconda3/envs/rlinf-isaacsim-env/bin/python scripts/smoke_leisaac.py
Run (pick_orange, GUI on :110, hold open):
    DISPLAY=:110 /home/hlei/miniconda3/envs/rlinf-isaacsim-env/bin/python \\
        scripts/smoke_leisaac.py --env-id LeIsaac-SO101-PickOrange-v0 --gui --hold
"""
import argparse
import os
import sys
import pathlib

_p = argparse.ArgumentParser()
_p.add_argument("--env-id", default="LeIsaac-SO101-LiftCube-v0")
_p.add_argument("--gui", action="store_true")
_p.add_argument("--steps", type=int, default=10)
_p.add_argument("--hold", action="store_true",
                help="after stepping, keep GUI open until Ctrl-C")
_args = _p.parse_args()
sys.argv = sys.argv[:1]

# leisaac resolves ASSETS_ROOT via `git rev-parse --show-toplevel` → our repo
# root (= /home/hlei/robotic/lerobot-rlinf/assets) by default, but the scene
# USDs live under third_party/leisaac/assets/. Override before import.
_REPO = pathlib.Path(__file__).resolve().parents[1]
os.environ["LEISAAC_ASSETS_ROOT"] = str(_REPO / "third_party" / "leisaac" / "assets")

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=not _args.gui, enable_cameras=True)
app = app_launcher.app

import gymnasium as gym
import numpy as np
import cv2
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import leisaac  # noqa: F401 — triggers gym.register(LeIsaac-SO101-*)
from leisaac.devices.action_process import init_action_cfg

ENV_ID = _args.env_id
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
for i in range(_args.steps):
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
        _tag = ENV_ID.lower().replace("leisaac-so101-", "").replace("-v0", "")
        path = OUT / f"leisaac_{_tag}_{k}.png"
        cv2.imwrite(str(path), bgr)
        print(f"[SMOKE] saved {path}  shape={arr.shape}")

print("[SMOKE] OK")

if _args.hold and _args.gui:
    print("[SMOKE] holding GUI open — Ctrl-C to exit")
    try:
        while app.is_running():
            app.update()
    except KeyboardInterrupt:
        pass

env.close()
app.close()
