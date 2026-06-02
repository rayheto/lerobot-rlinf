"""Standalone Phase-2 eval driver: SFT'd Pi 0.5 + RLinf openpi → sponge env.

Single process, no Ray. Uses RLinf's `OpenPi0ForRLActionPrediction` + transforms
+ norm_stats path (so behavior matches what Phase-3 PPO would see), just without
the distributed orchestration that caused tonight's Vulkan/CVD/botocore mess.

Run (headless):
    /home/hlei/miniconda3/envs/rlinf-isaacsim-env/bin/python scripts/eval_pi05_sponge.py
Run (GUI on :110):
    DISPLAY=:110 /home/hlei/miniconda3/envs/rlinf-isaacsim-env/bin/python scripts/eval_pi05_sponge.py --gui

Outputs per-episode {final_xy_dist_to_goal, max_lift, terminated, truncated} +
overall success rate using thresholds (XY < 0.08m AND max_lift > 0.05m).
"""
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--gui", action="store_true")
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--episodes", type=int, default=20, help="total episodes (must be multiple of num-envs)")
parser.add_argument("--max-steps", type=int, default=450)
parser.add_argument(
    "--model-path",
    default="/home/hlei/robotic/lerobot-rlinf/outputs/sft_pi05_sponge/openpi_remapped",
)
parser.add_argument("--xy-thresh", type=float, default=0.08)
parser.add_argument("--lift-thresh", type=float, default=0.05)
args = parser.parse_args()

sys.argv = sys.argv[:1]
assert args.episodes % args.num_envs == 0, "episodes must be a multiple of num-envs"

# RLinf is a dev checkout, not pip-installed in the conda env. Add to path
# before importing rlinf.* / openpi.* (openpi also lives in RLinf/.venv).
sys.path.insert(0, "/home/hlei/RLinf")
sys.path.insert(0, "/home/hlei/RLinf/.venv/lib/python3.11/site-packages")

# --- 1. Boot Kit BEFORE importing torch / RLinf model code (Kit owns pxr etc.) ---
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=not args.gui, enable_cameras=True)
simulation_app = app_launcher.app

import functools
print = functools.partial(print, flush=True)  # noqa: A001

import gymnasium as gym
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from omegaconf import OmegaConf

import lerobot_rlinf.tasks  # noqa: F401 — registers Isaac-Lift-Sponge-Bowl-SO101-Play-v0
from rlinf.models.embodiment.openpi import get_model

DEVICE = "cuda:0"
ENV_ID = "Isaac-Lift-Sponge-Bowl-SO101-Play-v0"
TASK_PROMPT = "Pick the blue sponge and place it in the bowl"


# --- 2. Build env ---
env_cfg = parse_env_cfg(ENV_ID, device=DEVICE, num_envs=args.num_envs)
# Default lift env truncates at 5s (250 steps). For long visual debugging or
# longer rollouts, let --max-steps win.
env_cfg.episode_length_s = max(env_cfg.episode_length_s, args.max_steps * 0.02 + 1.0)
env = gym.make(ENV_ID, cfg=env_cfg, render_mode="rgb_array").unwrapped
print(f"[EVAL] env ready  action_space={env.action_space.shape}  num_envs={args.num_envs}")


# --- 3. Build model via RLinf's get_model (uses safetensors + norm_stats on disk) ---
cfg = OmegaConf.create({
    "model_path": args.model_path,
    "model_type": "openpi",
    "action_dim": 6,
    "num_action_chunks": 5,
    "num_steps": 5,
    "add_value_head": True,
    "precision": "bfloat16",
    "openpi": {
        "config_name": "pi05_isaaclab_so101_lift",
        "num_images_in_input": 2,
        "noise_level": 0.5,
        "joint_logprob": False,
        "num_steps": 5,
        "value_after_vlm": True,
        "value_vlm_mode": "mean_token",
        "detach_critic_input": True,
        "action_chunk": 5,
        "action_dim": 32,        # padded inside model; env_dim is set below
        "action_env_dim": 6,
        "add_value_head": True,
    },
})
print("[EVAL] loading model …")
model = get_model(cfg).to(DEVICE).eval()
print("[EVAL] model loaded")


# --- 4. Eval loop ---
def wrap_obs(raw_obs, num_envs):
    return {
        "main_images": raw_obs["images"]["overhead"],   # [B, 224, 224, 3] uint8
        "wrist_images": raw_obs["images"]["wrist"],
        "states": raw_obs["policy"],                    # [B, 6] degrees
        "task_descriptions": [TASK_PROMPT] * num_envs,
        "extra_view_images": None,
    }


n_batches = args.episodes // args.num_envs
all_successes: list[bool] = []
all_finals: list[dict] = []

with torch.inference_mode():
    for batch in range(n_batches):
        raw_obs, _ = env.reset()
        env_obs = wrap_obs(raw_obs, args.num_envs)
        done = torch.zeros(args.num_envs, dtype=torch.bool, device=DEVICE)
        max_lift = torch.full((args.num_envs,), -float("inf"), device=DEVICE)
        final_xy_dist = torch.full((args.num_envs,), float("inf"), device=DEVICE)

        # Object / goal tracking via env scene handles (root_pos_w is [B, 3]).
        obj = env.scene["object"]
        goal_pos = torch.stack([
            torch.full((args.num_envs,), 0.30, device=DEVICE),  # bowl center x
            torch.full((args.num_envs,), -0.20, device=DEVICE), # bowl center y
        ], dim=1)

        step = 0
        chunk_idx = 0
        while step < args.max_steps and not done.all():
            # Pi 0.5 sample → action chunk [B, chunk*action_env_dim] = [B, 30]
            actions, _ = model.predict_action_batch(env_obs, mode="eval", compute_values=False)
            actions = actions.view(args.num_envs, 5, 6)  # [B, chunk, 6]

            if batch == 0 and chunk_idx < 3:
                print(f"[DIAG] chunk {chunk_idx}  env0 state pre-chunk = {env_obs['states'][0].tolist()}")
                for t_dbg in range(actions.shape[1]):
                    print(f"[DIAG]   action[env0, t={t_dbg}] = {actions[0, t_dbg].tolist()}")
            chunk_idx += 1

            for t in range(actions.shape[1]):
                if step >= args.max_steps or done.all():
                    break
                a_t = actions[:, t, :].to(DEVICE)
                raw_obs, _rew, term, trunc, _info = env.step(a_t)
                if batch == 0 and chunk_idx <= 3:
                    print(
                        f"[DIAG]   post-step state env0 = {raw_obs['policy'][0].tolist()}  "
                        f"(commanded = {a_t[0].tolist()})"
                    )

                # Track object lift + xy-distance to bowl goal.
                obj_pos = obj.data.root_pos_w[:, :3] - env.scene.env_origins
                lift_z = obj_pos[:, 2]
                xy_d = torch.linalg.norm(obj_pos[:, :2] - goal_pos, dim=1)
                # Only update for envs not yet done.
                active = ~done
                max_lift = torch.where(active & (lift_z > max_lift), lift_z, max_lift)
                final_xy_dist = torch.where(active, xy_d, final_xy_dist)

                done = done | term | trunc
                step += 1

            env_obs = wrap_obs(raw_obs, args.num_envs)

        success = (final_xy_dist < args.xy_thresh) & (max_lift > args.lift_thresh)
        for i in range(args.num_envs):
            all_successes.append(bool(success[i].item()))
            all_finals.append({
                "xy_dist": float(final_xy_dist[i].item()),
                "max_lift": float(max_lift[i].item()),
                "steps": step,
            })
            print(
                f"[EVAL] ep {batch * args.num_envs + i:3d}  "
                f"success={int(success[i].item())}  "
                f"xy_dist={final_xy_dist[i].item():.3f}m  "
                f"max_lift={max_lift[i].item():+.3f}m  "
                f"steps={step}"
            )

n_succ = sum(all_successes)
print(f"\n[EVAL] success rate: {n_succ}/{len(all_successes)} = {n_succ / len(all_successes):.1%}")
print(f"[EVAL] thresholds: xy<{args.xy_thresh}m, lift>{args.lift_thresh}m")

if args.gui:
    print("[EVAL] rollout done — GUI staying open. Ctrl-C to exit.")
    try:
        while simulation_app.is_running():
            simulation_app.update()
    except KeyboardInterrupt:
        pass

env.close()
simulation_app.close()
