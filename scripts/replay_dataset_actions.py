"""Replay one episode of dataset actions through Isaac-Lift-Sponge-Bowl-SO101 env.

Purpose: independent of any policy, verify dataset action units match env
action units. Both should be degrees in SO-101 URDF joint order. If they
diverge, env's JointPositionActionCfg scale/offset is wrong and SFT/RL
will silently fail downstream.

Pass: env joint trajectory tracks dataset state trajectory; per-step
joint-pos error stays < 5° (modulo gravity sag on a few joints).

Reads action + observation.state straight from the HF-cached parquet
shards. Bypasses LeRobotDataset on purpose: its current __getitem__
forces torchcodec video decoding (broken without system ffmpeg) and the
episode_data_index attribute has been removed. Episode boundaries come
from meta/episodes/.../file-000.parquet (dataset_from_index/_to_index).

Run (GUI):
    DISPLAY=:110 /home/hlei/miniconda3/envs/rlinf-isaacsim-env/bin/python \\
        scripts/replay_dataset_actions.py --gui --episode 0
Run (headless, prints stats only):
    /home/hlei/miniconda3/envs/rlinf-isaacsim-env/bin/python \\
        scripts/replay_dataset_actions.py --episode 0
"""
import argparse
import json
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--gui", action="store_true")
parser.add_argument("--episode", type=int, default=0)
parser.add_argument("--max-steps", type=int, default=0, help="0 = full episode")
parser.add_argument(
    "--warmup",
    type=int,
    default=30,
    help="Frames to skip when computing pass/fail stats (env spawns at zero "
    "pose; first ~30 frames are PD swinging to dataset's first-frame pose).",
)
parser.add_argument("--dump-json", type=str, default="")
parser.add_argument(
    "--dataset",
    default="aswinkumar99/LeRobot-SO101-task1-single-sponge-no-distractors-random-locations",
)
parser.add_argument(
    "--cache-root",
    default=os.path.expanduser("~/.cache/huggingface/lerobot"),
    help="HF_LEROBOT_HOME (where lerobot-train caches datasets).",
)
args = parser.parse_args()

sys.argv = sys.argv[:1]

# Read parquet BEFORE booting Isaac Sim — fail-fast on data issues without
# paying the multi-second AppLauncher cost.
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

ds_root = Path(args.cache_root) / args.dataset
ep_pq = ds_root / "meta/episodes/chunk-000/file-000.parquet"
data_pq = ds_root / "data/chunk-000/file-000.parquet"
for p in (ep_pq, data_pq):
    if not p.is_file():
        raise FileNotFoundError(f"missing parquet shard: {p}")

ep_table = pq.read_table(ep_pq, columns=["episode_index", "dataset_from_index", "dataset_to_index"])
ep_idx_arr = ep_table["episode_index"].to_numpy()
hits = np.flatnonzero(ep_idx_arr == args.episode)
if hits.size == 0:
    raise ValueError(f"episode {args.episode} not in {ep_pq}")
row = int(hits[0])
ep_from = ep_table["dataset_from_index"][row].as_py()
ep_to = ep_table["dataset_to_index"][row].as_py()
if args.max_steps > 0:
    ep_to = min(ep_to, ep_from + args.max_steps)

data_table = pq.read_table(data_pq, columns=["action", "observation.state", "index"])
mask = (data_table["index"].to_numpy() >= ep_from) & (data_table["index"].to_numpy() < ep_to)
ep_actions_np = np.stack(data_table["action"].to_numpy()[mask]).astype(np.float32)
ep_states_np = np.stack(data_table["observation.state"].to_numpy()[mask]).astype(np.float32)
T = ep_actions_np.shape[0]
print(f"[REPLAY] dataset {args.dataset} ep={args.episode}: {T} frames "
      f"({ep_from}..{ep_to})", flush=True)
print(f"[REPLAY] action[0]={ep_actions_np[0].tolist()}", flush=True)
print(f"[REPLAY] state [0]={ep_states_np[0].tolist()}", flush=True)

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=not args.gui, enable_cameras=True)
simulation_app = app_launcher.app

import functools  # noqa: E402
print = functools.partial(print, flush=True)  # noqa: A001 — Kit may close stdout late

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import lerobot_rlinf.tasks  # noqa: F401, E402

device = "cuda:0" if torch.cuda.is_available() else "cpu"
ep_actions = torch.from_numpy(ep_actions_np).to(device)
ep_states = torch.from_numpy(ep_states_np).to(device)

env_cfg = parse_env_cfg("Isaac-Lift-Sponge-Bowl-SO101-Play-v0", device=device, num_envs=1)
env = gym.make("Isaac-Lift-Sponge-Bowl-SO101-Play-v0", cfg=env_cfg)

obs, _ = env.reset()
print(f"[REPLAY] env reset, state[0]={obs['policy'][0].tolist()}")

# Per-step joint pos errors: |env_state - dataset_state| in degrees.
err_log = []
env_traj = []
with torch.inference_mode():
    for t in range(T):
        action = ep_actions[t : t + 1]  # [1, 6]
        obs, _r, _term, _trunc, _info = env.step(action)
        env_state = obs["policy"][0]
        ds_state = ep_states[t]
        err = (env_state - ds_state).abs()
        err_log.append(err.cpu().tolist())
        env_traj.append(env_state.cpu().tolist())
        if t % 50 == 0 or t == T - 1:
            print(
                f"[REPLAY] t={t:4d}  env={[f'{v:+7.2f}' for v in env_state.tolist()]}  "
                f"|err|max={err.max().item():.2f}°  |err|mean={err.mean().item():.2f}°"
            )

err_tensor = torch.tensor(err_log)  # [T, 6]
print()
warmup = min(args.warmup, T)
stable = err_tensor[warmup:]
print(f"[REPLAY] (raw, all {T} frames)")
print(f"[REPLAY]   per-joint mean |err|°: {err_tensor.mean(0).tolist()}")
print(f"[REPLAY]   overall  mean |err|°: {err_tensor.mean().item():.2f}  "
      f"max |err|°: {err_tensor.max().item():.2f}")
print(f"[REPLAY] (stable, skip first {warmup} warmup frames)")
print(f"[REPLAY]   per-joint mean |err|°: {stable.mean(0).tolist()}")
print(f"[REPLAY]   per-joint  max |err|°: {stable.max(0).values.tolist()}")
print(f"[REPLAY]   overall  mean |err|°: {stable.mean().item():.2f}  "
      f"max |err|°: {stable.max().item():.2f}")

# Action units match if stable mean < 5°. First N frames are env→dataset
# PD swing, not a unit-match signal — they always look bad regardless.
if stable.mean() < 5.0:
    print(f"[REPLAY] PASS — action units align (stable mean err {stable.mean().item():.2f}° < 5°)")
else:
    print(f"[REPLAY] FAIL — stable mean err {stable.mean().item():.2f}° >= 5°; "
          "inspect JointPositionActionCfg scale/offset")

if args.dump_json:
    out = {
        "episode": args.episode,
        "T": T,
        "dataset_actions": ep_actions.cpu().tolist(),
        "dataset_states": ep_states.cpu().tolist(),
        "env_states": env_traj,
        "per_step_err_deg": err_log,
    }
    Path(args.dump_json).write_text(json.dumps(out))
    print(f"[REPLAY] dumped traj to {args.dump_json}")

if args.gui:
    print("[REPLAY] holding last action — Ctrl+C to exit")
    try:
        with torch.inference_mode():
            last = ep_actions[-1:].clone()
            while simulation_app.is_running():
                env.step(last)
    except KeyboardInterrupt:
        pass

env.close()
simulation_app.close()
