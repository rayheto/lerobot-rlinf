"""Smoke test: openpi model (with weight-remap + IMAGE_KEYS fixes) on pick_orange.

Verifies that the repaired openpi path produces correct actions before we
wire it into the PPO runner.  Compares against the known-good lerobot-native
adapter baseline (80% grasp rate on 5-ep test).

Usage:
    /home/hlei/miniconda3/envs/rlinf-isaacsim-env/bin/python \
        scripts/eval_pi05_pickorange_openpi.py \
        --model-path outputs/sft_pi05_pickorange/checkpoints/034000/pretrained_model \
        --num-envs 1 --episodes 5 --max-steps 1500
"""
import argparse
import os
import pathlib
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--gui", action="store_true")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--max-steps", type=int, default=1500)
parser.add_argument("--model-path",
                    default="/home/hlei/robotic/lerobot-rlinf/outputs/sft_pi05_pickorange/openpi_remapped")
parser.add_argument("--config-name", default="pi05_isaaclab_so101_pick_orange")
args = parser.parse_args()

sys.argv = sys.argv[:1]
assert args.episodes % args.num_envs == 0

_REPO = pathlib.Path(__file__).resolve().parents[1]
os.environ["LEISAAC_ASSETS_ROOT"] = str(_REPO / "third_party" / "leisaac" / "assets")

sys.path.insert(0, "/home/hlei/RLinf")
sys.path.insert(0, "/home/hlei/RLinf/.venv/lib/python3.11/site-packages")

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=not args.gui, enable_cameras=True)
simulation_app = app_launcher.app

import functools
print = functools.partial(print, flush=True)

import gymnasium as gym
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from omegaconf import OmegaConf

import leisaac  # noqa: F401
from leisaac.devices.action_process import init_action_cfg
from leisaac.utils.robot_utils import (
    convert_leisaac_action_to_lerobot,
    convert_lerobot_action_to_leisaac,
    is_so101_at_rest_pose,
)
from rlinf.models.embodiment.openpi import get_model

DEVICE = "cuda:0"
ENV_ID = "LeIsaac-SO101-PickOrange-v0"
TASK_PROMPT = "Grab orange and place into plate"

# --- env ---
env_cfg = parse_env_cfg(ENV_ID, device=DEVICE, num_envs=args.num_envs)
init_action_cfg(env_cfg.actions, device="so101leader")
env_cfg.episode_length_s = max(env_cfg.episode_length_s, args.max_steps * 0.02 + 1.0)
env = gym.make(ENV_ID, cfg=env_cfg, render_mode="rgb_array").unwrapped
print(f"[EVAL] env ready  action_space={env.action_space.shape}  num_envs={args.num_envs}")

ROBOT_JOINT_NAMES = list(env.scene["robot"].data.joint_names)

# --- model ---
cfg = OmegaConf.create({
    "model_path": args.model_path,
    "model_type": "openpi",
    "action_dim": 6,
    "num_action_chunks": 50,
    "num_steps": 10,
    "add_value_head": True,
    "precision": "bfloat16",
    "openpi": {
        "config_name": args.config_name,
        "discrete_state_input": True,
        "num_images_in_input": 2,
        "noise_level": 0.5,
        "joint_logprob": False,
        "num_steps": 10,
        "value_after_vlm": True,
        "value_vlm_mode": "mean_token",
        "detach_critic_input": True,
        "action_chunk": 50,
        "action_dim": 32,
        "action_env_dim": 6,
        "add_value_head": True,
    },
})
print("[EVAL] loading openpi model …")
model = get_model(cfg).to(DEVICE).eval()
print("[EVAL] model loaded")


def wrap_obs(raw_obs, num_envs):
    policy = raw_obs["policy"]
    front = policy["front"]
    wrist = policy["wrist"]
    joint_pos = policy["joint_pos"]
    states_motor_deg_np = convert_leisaac_action_to_lerobot(joint_pos)
    states = torch.from_numpy(states_motor_deg_np).float().to(joint_pos.device)
    return {
        "main_images": front,
        "wrist_images": wrist,
        "states": states,
        "task_descriptions": [TASK_PROMPT] * num_envs,
        "extra_view_images": None,
    }


n_batches = args.episodes // args.num_envs
all_successes = []
diag_dumped = 0

with torch.inference_mode():
    for batch in range(n_batches):
        raw_obs, _ = env.reset()
        env_obs = wrap_obs(raw_obs, args.num_envs)
        done = torch.zeros(args.num_envs, dtype=torch.bool, device=DEVICE)
        succ = torch.zeros(args.num_envs, dtype=torch.bool, device=DEVICE)

        n_oranges = 3
        ever_picked = torch.zeros(args.num_envs, n_oranges, dtype=torch.bool, device=DEVICE)
        ever_placed = torch.zeros(args.num_envs, n_oranges, dtype=torch.bool, device=DEVICE)

        step = 0
        while step < args.max_steps and not done.all():
            actions, _ = model.predict_action_batch(env_obs, mode="eval", compute_values=False)
            actions = actions.view(args.num_envs, 50, 6)

            for t in range(50):
                if done.all():
                    break
                a_t = actions[:, t, :]
                a_env_np = convert_lerobot_action_to_leisaac(a_t)
                a_env = torch.from_numpy(a_env_np).float().to(DEVICE)

                if diag_dumped < 5 and step == 0 and t < 5:
                    print(f"[DIAG] t={t} action_motor_deg[0]: {[round(float(x), 2) for x in a_t[0]]}")
                    if t == 4:
                        diag_dumped += 1

                raw_obs, reward, term, trunc, _ = env.step(a_env)
                done = done | torch.as_tensor(term | trunc).bool().to(DEVICE)

                sub = raw_obs.get("subtask_terms")
                if sub is not None:
                    for i in range(1, n_oranges + 1):
                        pk = sub.get(f"pick_orange00{i}")
                        if pk is not None:
                            ever_picked[:, i - 1] |= torch.as_tensor(pk).bool().to(DEVICE)
                        pl = sub.get(f"put_orange00{i}_to_plate")
                        if pl is not None:
                            ever_placed[:, i - 1] |= torch.as_tensor(pl).bool().to(DEVICE)

                term_t = torch.as_tensor(term).bool().to(DEVICE)
                trunc_t = torch.as_tensor(trunc).bool().to(DEVICE)
                succ |= ~done & term_t & ~trunc_t

                if done.all():
                    break

                env_obs = wrap_obs(raw_obs, args.num_envs)
                step += 1

        success_count = int(succ.sum())
        all_successes.append(success_count)
        picked_str = str([bool(ever_picked[0, i].item()) for i in range(n_oranges)])
        placed_str = str([bool(ever_placed[0, i].item()) for i in range(n_oranges)])
        print(f"[EVAL] ep {batch:3d}  success={success_count}  steps={step}  "
              f"picked={int(ever_picked[0].sum())}/3 {picked_str}  "
              f"placed={int(ever_placed[0].sum())}/3 {placed_str}")

total_success = sum(all_successes)
total_episodes = len(all_successes) * args.num_envs
print(f"\n[EVAL] success rate: {total_success}/{total_episodes} = {total_success/total_episodes*100:.1f}%")
