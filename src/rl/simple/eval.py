"""Offline eval for a Step-8 PPO ckpt.

Loads the trained residual head + value head from `head_iter*.pt`, drives the
Isaac Lab pick-orange env with a JAX pi05 server, runs N episodes with the
mean-action (deterministic) residual policy, and reports:

  * total success rate over all completed episodes
  * "fast" success rate (episode length <= fast_steps)
  * length histogram across success buckets

Run:
  python -m src.rl.simple.eval \\
      --ckpt logs/simple_ppo_step8f_jax/ckpts/head_iter000050.pt \\
      --out  logs/eval_step8f --n-episodes 60 --num-envs 2

Success = `success_once` (env's rest_emitted: all 3 oranges placed + arm rest).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import time
from typing import List, Optional

import torch

from src.rl.simple._openpi_server import kill_server
from src.rl.simple.config import Config
from src.rl.simple.policy import ResidualGaussianPolicy
from src.rl.simple.train import _build_env, _capture_frame, _start_openpi_server


@torch.no_grad()
def _act_mean(policy: ResidualGaussianPolicy, obs, reset_mask):
    """Deterministic action: pi05 chunk slot + residual mean (no σ noise)."""
    base = policy.pi05.base_action(obs, reset_mask=reset_mask)  # (Nenv, 6)
    states = obs["states"].to(policy.device, torch.float32)
    h = policy._head_input(states, base)
    dist = policy._dist_from_head(h)
    action = base + dist.mean
    return action


def evaluate(
    ckpt_path: pathlib.Path,
    out_dir: pathlib.Path,
    n_episodes: int,
    num_envs: int,
    max_ep_steps: int,
    fast_steps: int,
    episode_length_s: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a Config override for eval.
    cfg = Config()
    cfg.env_num_envs = num_envs
    cfg.env_max_episode_steps = max_ep_steps
    cfg.env_episode_length_s = episode_length_s
    cfg.rollout_len = max_ep_steps  # used by env wrapper
    cfg.log_dir = str(out_dir)
    cfg.deterministic_pi05 = True

    server_proc = _start_openpi_server(cfg)
    log_path = out_dir / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    log_fh = log_path.open("w")
    vdir = out_dir / "video"
    vdir.mkdir(parents=True, exist_ok=True)
    import imageio.v3 as iio
    print(f"[eval] writing jsonl to {log_path}", flush=True)
    print(f"[eval] writing per-episode env-0 videos to {vdir}", flush=True)

    try:
        env = _build_env(cfg)
        print(f"[eval] env built. num_envs={env.num_envs} device={env.device} "
              f"max_steps={max_ep_steps}", flush=True)

        policy = ResidualGaussianPolicy(cfg, device=cfg.device)
        state = torch.load(ckpt_path, map_location=cfg.device, weights_only=True)
        policy.head.load_state_dict(state["head"])
        policy.value_head.load_state_dict(state["value_head"])
        policy.head.eval()
        policy.value_head.eval()
        print(f"[eval] loaded ckpt {ckpt_path} (iter={state.get('iter')})", flush=True)

        records: List[dict] = []
        ep_steps = torch.zeros(num_envs, device=cfg.device)
        reset_mask = torch.ones(num_envs, dtype=torch.bool, device=cfg.device)
        obs, _ = env.reset(seed=cfg.env_seed)
        ep0_video_frames: list = []
        ep0_video_frames.append(_capture_frame(obs))

        max_global_steps = max_ep_steps * (n_episodes // num_envs + 4)
        for t in range(max_global_steps):
            action = _act_mean(policy, obs, reset_mask=reset_mask)
            obs, _, term, trunc, info = env.step(action, auto_reset=True)
            done = (term | trunc).to(cfg.device)
            reset_mask = done
            ep_steps += 1.0
            ep0_video_frames.append(_capture_frame(obs))

            if done.any():
                ep_info = info.get("final_info", {}).get("episode", {})
                if not ep_info:
                    ep_info = info.get("episode", {})
                succ = ep_info.get("success_once")
                failA = ep_info.get("failA_once")
                done_idx = torch.nonzero(done, as_tuple=False).flatten().tolist()
                for i in done_idx:
                    steps_i = int(ep_steps[i].item())
                    s = bool(succ[i].item()) if succ is not None else False
                    fA = bool(failA[i].item()) if failA is not None else False
                    rec = {
                        "ep": len(records),
                        "env": i,
                        "steps": steps_i,
                        "success": s,
                        "failA": fA,
                        "fast": s and steps_i <= fast_steps,
                    }
                    records.append(rec)
                    log_fh.write(json.dumps(rec) + "\n")
                    log_fh.flush()
                    ep_steps[i] = 0.0
                    if i == 0 and ep0_video_frames:
                        tag = "succ" if s else ("failA" if fA else "fail")
                        vpath = vdir / f"ep{rec['ep']:03d}_env0_{tag}_steps{steps_i:04d}.mp4"
                        try:
                            iio.imwrite(str(vpath), ep0_video_frames, fps=cfg.video_fps, codec="libx264")
                            print(f"[eval] wrote video {vpath}", flush=True)
                        except Exception as e:
                            print(f"[eval] video write failed: {e}", flush=True)
                        ep0_video_frames = []
                    print(
                        f"[eval] ep#{rec['ep']:3d} env{i} steps={steps_i:4d} "
                        f"succ={s} fast={rec['fast']} failA={fA}",
                        flush=True,
                    )
                if len(records) >= n_episodes:
                    break

        # Summary.
        N = len(records)
        n_succ = sum(r["success"] for r in records)
        n_fast = sum(r["fast"] for r in records)
        n_failA = sum(r["failA"] for r in records)
        len_buckets = {"<=900": 0, "901-1800": 0, "1801-2700": 0}
        for r in records:
            s = r["steps"]
            if s <= 900: len_buckets["<=900"] += 1
            elif s <= 1800: len_buckets["901-1800"] += 1
            else: len_buckets["1801-2700"] += 1

        summary = {
            "event": "summary",
            "ckpt": str(ckpt_path),
            "n_episodes": N,
            "success_rate": n_succ / N if N else 0.0,
            "fast_success_rate": n_fast / N if N else 0.0,
            "n_success": n_succ,
            "n_fast_success": n_fast,
            "n_failA": n_failA,
            "fast_steps_threshold": fast_steps,
            "len_buckets_all": len_buckets,
        }
        log_fh.write(json.dumps(summary) + "\n")
        print("=" * 60, flush=True)
        print(f"[eval] SUMMARY  ({N} episodes)", flush=True)
        print(f"  ckpt:             {ckpt_path}", flush=True)
        print(f"  success_rate:     {summary['success_rate']*100:.2f}%  ({n_succ}/{N})", flush=True)
        print(f"  fast_success(<={fast_steps}): {summary['fast_success_rate']*100:.2f}%  ({n_fast}/{N})", flush=True)
        print(f"  n_failA:          {n_failA}", flush=True)
        print(f"  len buckets:      {len_buckets}", flush=True)
        print("=" * 60, flush=True)
    finally:
        log_fh.close()
        if server_proc is not None:
            kill_server(server_proc)
        print("[eval] done.", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--out", required=True, type=str)
    p.add_argument("--n-episodes", type=int, default=60)
    p.add_argument("--num-envs", type=int, default=2)
    p.add_argument("--max-ep-steps", type=int, default=2700, help="outer-loop safety net (60Hz inner)")
    p.add_argument("--fast-steps", type=int, default=900, help="fast-success bucket cap (15s @ 60Hz)")
    p.add_argument("--episode-length-s", type=float, default=45.0,
                   help="leisaac internal trunc seconds (×60Hz = inner step cap). 45s=2700 step")
    args = p.parse_args()
    evaluate(
        ckpt_path=pathlib.Path(args.ckpt).expanduser(),
        out_dir=pathlib.Path(args.out).expanduser(),
        n_episodes=args.n_episodes,
        num_envs=args.num_envs,
        max_ep_steps=args.max_ep_steps,
        fast_steps=args.fast_steps,
        episode_length_s=args.episode_length_s,
    )


if __name__ == "__main__":
    main()
