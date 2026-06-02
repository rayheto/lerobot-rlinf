"""Smoke-test the Isaac-Lift-Cube-SO101-v0 env: reset + a few zero-action steps.

Run (headless):
    /home/hlei/miniconda3/envs/rlinf-isaacsim-env/bin/python scripts/smoke_lift_so101.py
Run (GUI on display :110):
    DISPLAY=:110 /home/hlei/miniconda3/envs/rlinf-isaacsim-env/bin/python scripts/smoke_lift_so101.py --gui
"""
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--gui", action="store_true")
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=20)
args = parser.parse_args()

sys.argv = sys.argv[:1]

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=not args.gui, enable_cameras=True)
simulation_app = app_launcher.app

import functools
print = functools.partial(print, flush=True)  # noqa: A001 — Kit may close stdout late; force flush

import gymnasium as gym
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import lerobot_rlinf.tasks  # noqa: F401  (registers Isaac-Lift-Cube-SO101-v0)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
env_cfg = parse_env_cfg(
    "Isaac-Lift-Cube-SO101-Play-v0",
    device=device,
    num_envs=args.num_envs,
)
env = gym.make("Isaac-Lift-Cube-SO101-Play-v0", cfg=env_cfg)
print(f"[SMOKE] action_space={env.action_space}")
print(f"[SMOKE] observation_space={env.observation_space}")

obs, _ = env.reset()
state = obs["policy"] if isinstance(obs, dict) else obs
print(f"[SMOKE] reset OK, observation.state shape={tuple(state.shape)} (expect [B, 6] degrees)")
print(f"[SMOKE] state row 0 = {state[0].tolist()}  (expect arm joints ~0°, gripper ~45.8°)")

if isinstance(obs, dict) and "images" in obs:
    for name, tensor in obs["images"].items():
        print(
            f"[SMOKE] image '{name}' shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"min={int(tensor.min())} max={int(tensor.max())}"
        )

# Action is [B, 6] in degrees, URDF joint order:
# [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper].
# Zero action = 0° on every joint → URDF mechanical zero pose.
action_dim = env.action_space.shape[1] if len(env.action_space.shape) > 1 else env.action_space.shape[0]
assert action_dim == 6, f"expected action_dim=6 (SO-101), got {action_dim}"

home_action = torch.zeros((args.num_envs, action_dim), device=device)
with torch.inference_mode():
    for i in range(args.steps):
        obs, rew, term, trunc, info = env.step(home_action)
        if i % 5 == 0:
            print(f"[SMOKE] step {i:3d}  rew_mean={rew.float().mean().item():+.4f}  term={int(term.sum())}  trunc={int(trunc.sum())}")

# State should be near 0° per joint after holding zero command (modulo gravity sag).
final_state = obs["policy"]
print(f"[SMOKE] final state row 0 = {final_state[0].tolist()}  (expect arm ~0°, gripper ~0°)")

print("[SMOKE] OK")

if args.gui:
    print("[SMOKE] idle loop with small random actions — Ctrl+C to exit")
    try:
        with torch.inference_mode():
            while simulation_app.is_running():
                # Small random walk in Feetech-norm units: ±5% per joint per step.
                action = (torch.rand((args.num_envs, action_dim), device=device) - 0.5) * 10.0
                env.step(action)
    except KeyboardInterrupt:
        pass

env.close()
simulation_app.close()
