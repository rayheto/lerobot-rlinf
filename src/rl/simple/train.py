"""Simple PPO training entrypoint for SO-101 PickOrange residual policy.

Single-process, single-GPU. No Ray / FSDP / hydra.

Usage:
    python -m src.rl.simple.train                          # full training
    python -m src.rl.simple.train --dryrun-env             # 100 random-action env steps
    python -m src.rl.simple.train --total-iters 4 --rollout-len 8

For the policy-only smoke test (no env, no training):
    python -m src.rl.simple.policy --smoke
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import time
from typing import Any, Optional

import torch

from src.rl.simple.bc_anchor import BCAnchor
from src.rl.simple.config import Config
from src.rl.simple._openpi_server import (
    kill_server,
    spawn_openpi_server,
    wait_for_port,
)
from src.rl.simple.policy import ResidualGaussianPolicy
from src.rl.simple.ppo import PPOTrainer
from src.rl.simple.reward_shaping import ShapedReward
from src.rl.simple.rollout_buffer import RolloutBuffer

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _resolve_path(p: str) -> pathlib.Path:
    pp = pathlib.Path(p)
    if not pp.is_absolute():
        pp = _REPO_ROOT / pp
    return pp


def _start_openpi_server(cfg: Config):
    """Spawn the JAX serve_policy.py process and wait for it to listen.

    Returns the Popen handle so the caller can kill it on exit. If a server
    already happens to be listening on the configured port, we DON'T spawn —
    that case is convenient during interactive debugging.
    """
    import socket as _socket
    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect((cfg.pi05_server_host, cfg.pi05_server_port))
            print(
                f"[openpi-server] reusing existing listener at "
                f"{cfg.pi05_server_host}:{cfg.pi05_server_port}",
                flush=True,
            )
            return None
        except OSError:
            pass

    ckpt = _resolve_path(cfg.pi05_jax_ckpt_dir)
    if not ckpt.is_dir():
        raise SystemExit(f"JAX ckpt dir missing: {ckpt}")
    if not (ckpt / "params").is_dir():
        raise SystemExit(f"ckpt is not JAX/Orbax format (missing params/): {ckpt}")
    proc = spawn_openpi_server(
        ckpt=ckpt,
        config_name=cfg.pi05_jax_config_name,
        prompt=cfg.env_prompt,
        port=cfg.pi05_server_port,
    )
    try:
        wait_for_port(cfg.pi05_server_host, cfg.pi05_server_port, cfg.pi05_server_startup_s)
        time.sleep(3.0)  # extra grace for websocket handshake
    except SystemExit:
        kill_server(proc)
        raise
    return proc


# ---------------------------------------------------------------------------
# CLI overrides
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dryrun-env", action="store_true",
                   help="Drive env with random actions for a few steps; no training.")
    # Field overrides — `from __future__ import annotations` mak          es f.type a
    # str, so resolve by reading the default's runtime type.
    _SCALAR = (int, float, str, bool)
    for f in dataclasses.fields(Config):
        t = type(f.default) if f.default is not dataclasses.MISSING else None
        if t not in _SCALAR:
            continue
        kw: dict[str, Any] = {"default": None}
        if t is bool:
            kw["action"] = "store_true"
        else:
            kw["type"] = t
        p.add_argument(f"--{f.name.replace('_', '-')}", **kw)
    return p


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    for f in dataclasses.fields(Config):
        v = getattr(args, f.name, None)
        if v is None or v is False:
            continue
        setattr(cfg, f.name, v)
    return cfg


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class JsonlLogger:
    def __init__(self, log_dir: str):
        d = pathlib.Path(log_dir).expanduser()
        d.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.path = d / f"train_{ts}.jsonl"
        self._fh = self.path.open("w")
        print(f"[log] writing jsonl to {self.path}", flush=True)
        # tensorboard: one scalar per iter, every numeric metric
        self._tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tb = SummaryWriter(log_dir=str(d / "tb"))
            print(f"[log] tensorboard at {d}/tb", flush=True)
        except Exception as e:
            print(f"[log] tensorboard disabled: {e}", flush=True)

    def log(self, record: dict) -> None:
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()
        if self._tb is not None and "iter" in record:
            step = int(record["iter"])
            for k, v in record.items():
                if isinstance(v, (int, float)) and k != "iter":
                    self._tb.add_scalar(k, v, step)

    def close(self) -> None:
        self._fh.close()
        if self._tb is not None:
            self._tb.close()


# ---------------------------------------------------------------------------
# Video capture helper
# ---------------------------------------------------------------------------


def _capture_frame(obs):
    """Build a side-by-side (front | wrist) uint8 RGB frame from env-0 obs.

    Tolerates tensors or arrays, (H,W,C) or (C,H,W), float[0,1] or uint8.
    """
    import numpy as np

    def _to_hwc_uint8(x):
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if x.ndim == 4:
            x = x[0]
        if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
            x = np.transpose(x, (1, 2, 0))
        if x.dtype != np.uint8:
            xmax = float(x.max()) if x.size else 1.0
            if xmax <= 1.5:
                x = (x * 255.0).clip(0, 255).astype(np.uint8)
            else:
                x = x.clip(0, 255).astype(np.uint8)
        if x.ndim == 2:
            x = np.stack([x, x, x], axis=-1)
        if x.shape[-1] == 1:
            x = np.repeat(x, 3, axis=-1)
        return x

    front = _to_hwc_uint8(obs["main_images"])
    wrist = _to_hwc_uint8(obs["wrist_images"])
    if front.shape[0] != wrist.shape[0]:
        h = max(front.shape[0], wrist.shape[0])
        if front.shape[0] != h:
            pad = np.zeros((h - front.shape[0], front.shape[1], 3), dtype=np.uint8)
            front = np.concatenate([front, pad], axis=0)
        if wrist.shape[0] != h:
            pad = np.zeros((h - wrist.shape[0], wrist.shape[1], 3), dtype=np.uint8)
            wrist = np.concatenate([wrist, pad], axis=0)
    return np.concatenate([front, wrist], axis=1)


# ---------------------------------------------------------------------------
# Env build (single-process IsaaclabPickOrangeEnv)
# ---------------------------------------------------------------------------


def _setup_env_vars() -> None:
    """Mirror src/rl/run.sh:21-31 so the subprocess env finds leisaac assets
    and openpi/rlinf packages. Idempotent."""
    import os
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    leisaac_assets = repo_root / "third_party" / "leisaac" / "assets"
    os.environ.setdefault("LEISAAC_ASSETS_ROOT", str(leisaac_assets))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("RLINF_OPENPI_SO101_PATCH", "1")
    os.environ.setdefault("EMBODIED_PATH",
                          str(repo_root / "third_party" / "RLinf" / "examples" / "embodiment"))


def _build_env(cfg: Config):
    _setup_env_vars()
    from src.rl.envs.isaaclab_pick_orange import IsaaclabPickOrangeEnv
    env_cfg = cfg.build_env_cfg()
    env = IsaaclabPickOrangeEnv(
        env_cfg, num_envs=cfg.env_num_envs,
        seed_offset=0, total_num_processes=1, worker_info=None,
    )
    return env


# ---------------------------------------------------------------------------
# Dryrun: env wiring (random actions for N steps)
# ---------------------------------------------------------------------------


def _dryrun_env(cfg: Config, num_steps: int = 100, video: bool = True) -> None:
    env = _build_env(cfg)
    obs, _ = env.reset(seed=cfg.env_seed)
    # step_dt is inside the SubProc worker so we can't read it from main. Measure
    # wallclock per env.step over a small batch instead — at decimation=2 / sim
    # 60Hz the physics advances 1/30s sim per env.step (wallclock will be longer
    # because of GPU sync, but the ratio confirms decimation took effect).
    print(f"[dryrun-env] obs keys={list(obs.keys())} states.shape={tuple(obs['states'].shape)} "
          f"cfg.env_decimation={cfg.env_decimation} (expect 2 for 30Hz native)",
          flush=True)

    frames = [] if video else None
    if video:
        frames.append(_capture_frame(obs))

    for t in range(num_steps):
        action = torch.zeros(env.num_envs, 6, device=env.device, dtype=torch.float32)
        action += 0.01 * torch.randn_like(action)
        obs, r, term, trunc, info = env.step(action, auto_reset=True)
        if video:
            frames.append(_capture_frame(obs))
        if t % 10 == 0:
            print(f"[dryrun-env] t={t} r={r.cpu().tolist()} "
                  f"term={term.cpu().tolist()} trunc={trunc.cpu().tolist()}", flush=True)

    if video and frames:
        import imageio.v3 as iio
        out_dir = pathlib.Path(cfg.log_dir).expanduser() / "video"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"dryrun_env_{ts}.mp4"
        iio.imwrite(str(path), frames, fps=30, codec="libx264")
        print(f"[dryrun-env] wrote video: {path} ({len(frames)} frames, "
              f"size={frames[0].shape})", flush=True)
    print("[dryrun-env] OK", flush=True)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def _save_ckpt(policy: ResidualGaussianPolicy, log_dir: str, it: int) -> None:
    d = pathlib.Path(log_dir).expanduser() / "ckpts"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"head_iter{it:06d}.pt"
    torch.save({
        "iter": it,
        "head": policy.head.state_dict(),
        "value_head": policy.value_head.state_dict(),
    }, path)
    print(f"[ckpt] saved {path}", flush=True)


def train(cfg: Config) -> None:
    torch.manual_seed(cfg.seed)

    logger = JsonlLogger(cfg.log_dir)
    logger.log({"event": "config", **dataclasses.asdict(cfg)})

    server_proc = _start_openpi_server(cfg)
    try:
        _train_with_server(cfg, logger)
    finally:
        logger.close()
        if server_proc is not None:
            kill_server(server_proc)
        print("[train] done.", flush=True)


def _train_with_server(cfg: Config, logger: "JsonlLogger") -> None:
    env = _build_env(cfg)
    print(f"[train] env built. num_envs={env.num_envs} device={env.device}", flush=True)

    policy = ResidualGaussianPolicy(cfg, device=cfg.device)
    optim = torch.optim.AdamW(
        policy.trainable_params(),
        lr=cfg.lr, eps=cfg.adam_eps, weight_decay=cfg.weight_decay,
    )
    bc = BCAnchor(cfg.demo_dataset_path, batch_size=cfg.bc_batch_size, device=cfg.device)
    print(f"[train] bc anchor loaded: N={bc.N} demo pairs", flush=True)

    buffer = RolloutBuffer(
        T=cfg.rollout_len, Nenv=env.num_envs,
        state_dim=ResidualGaussianPolicy.STATE_DIM,
        act_dim=ResidualGaussianPolicy.ACT_DIM,
        device=cfg.device,
    )
    shaper = ShapedReward(cfg)
    trainer = PPOTrainer(policy, optim, cfg, bc_anchor=bc)

    obs, _ = env.reset(seed=cfg.env_seed)
    n_envs = env.num_envs
    # First call must force a fresh pi05 infer for every env — there is no
    # cached chunk yet. Subsequent steps pass the previous step's `done` mask
    # so any reset env re-infers (instead of replaying a stale chunk).
    reset_mask = torch.ones(n_envs, dtype=torch.bool, device=cfg.device)

    ep_return = torch.zeros(n_envs, device=cfg.device)
    ep_len = torch.zeros(n_envs, device=cfg.device)
    completed_returns: list[float] = []
    completed_lens: list[float] = []
    completed_successes: list[float] = []
    completed_fails: list[float] = []
    completed_failsA: list[float] = []

    # Video state — persistent across iters because an env-0 episode often
    # straddles an iter boundary. Each saved mp4 is exactly ONE complete env-0
    # episode (from fresh reset to next done), with a name reflecting the
    # global episode index (not the training iter).
    # video: dump one mp4 per completed env-0 episode (no iter gating).
    ep0_idx = 0
    ep0_video_frames: list = []

    for it in range(cfg.total_iters):
        t_iter = time.time()
        buffer.reset()
        rollout_r_info = {
            "mean_env_reward": 0.0,
            "mean_ood_penalty": 0.0,
            "mean_survival_cost": 0.0,
            "mean_dense_eo": 0.0,
            "mean_dense_lift": 0.0,
            "mean_shaped_reward": 0.0,
        }
        for t in range(cfg.rollout_len):
            with torch.no_grad():
                action, base, logp, value = policy.act(obs, reset_mask=reset_mask)
            state_at_step = obs["states"].to(cfg.device, torch.float32)
            next_obs, env_r, term, trunc, info = env.step(action, auto_reset=True)
            done = (term | trunc).to(cfg.device)
            # Next iteration must re-infer a fresh chunk for any env that just
            # reset (auto_reset returned new obs[i] from a fresh episode).
            reset_mask = done
            env_r = env_r.to(cfg.device, torch.float32)

            # Dense shaping needs POST-step aux (ee_pos, orange001_pos) and the
            # episode-start orange z. Both live on the env wrapper as plain
            # attributes; safe to read in-process.
            aux = getattr(env, "_last_aux", None)
            orange_init_z = getattr(env, "_orange_init_z", None)
            shaped, r_info = shaper(env_r, state_at_step, done, aux=aux, orange_init_z=orange_init_z)
            buffer.add(state_at_step, base, action, logp, value, shaped, done.float())
            for k in rollout_r_info:
                rollout_r_info[k] += r_info[k] / cfg.rollout_len

            ep_return += shaped
            ep_len += 1.0
            if done.any():
                done_idx = torch.nonzero(done, as_tuple=False).flatten().tolist()
                # RLinf's _handle_auto_reset stashes pre-reset `infos` under
                # `final_info` and OVERWRITES the top-level `infos` with the
                # fresh reset state. So success_once / fail_once for the
                # episode that just ended live in info["final_info"]["episode"],
                # NOT info["episode"]. The latter is from the post-reset call
                # and never has these flags.
                ep_info = info.get("final_info", {}).get("episode", {})
                if not ep_info:
                    ep_info = info.get("episode", {})  # fallback (no auto-reset)
                succ = ep_info.get("success_once")
                fail = ep_info.get("fail_once")
                failA = ep_info.get("failA_once")
                for i in done_idx:
                    completed_returns.append(float(ep_return[i].item()))
                    completed_lens.append(float(ep_len[i].item()))
                    if succ is not None:
                        completed_successes.append(float(succ[i].item()))
                    if fail is not None:
                        completed_fails.append(float(fail[i].item()))
                    if failA is not None:
                        completed_failsA.append(float(failA[i].item()))
                    ep_return[i] = 0.0
                    ep_len[i] = 0.0

            # ---- env-0 per-episode video: flush mp4 on every done ----
            env0_done = bool(done[0].item())
            if env0_done and ep0_video_frames:
                try:
                    import imageio.v3 as iio
                    vdir = pathlib.Path(cfg.log_dir).expanduser() / "video"
                    vdir.mkdir(parents=True, exist_ok=True)
                    vpath = vdir / f"episode_{ep0_idx:06d}_iter{it:03d}.mp4"
                    iio.imwrite(
                        str(vpath), ep0_video_frames,
                        fps=cfg.video_fps, codec="libx264",
                    )
                    print(
                        f"[video] ep0 #{ep0_idx} ({len(ep0_video_frames)} frames) -> {vpath}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[video] ep0 #{ep0_idx} save failed: {e}", flush=True)
                ep0_video_frames = []
                ep0_idx += 1

            obs = next_obs
            ep0_video_frames.append(_capture_frame(obs))

        with torch.no_grad():
            last_value = policy.value(obs)
        buffer.compute_gae(cfg.gamma, cfg.gae_lambda, last_value, normalize=True)

        metrics = trainer.update(buffer)

        t_iter = time.time() - t_iter
        record = {
            "event": "iter",
            "iter": it,
            "time_sec": t_iter,
            **rollout_r_info,
            **metrics,
        }
        if completed_returns:
            window = min(len(completed_returns), 32)
            record["ep_return_mean"] = sum(completed_returns[-window:]) / window
            record["ep_len_mean"] = sum(completed_lens[-window:]) / window
            if completed_successes:
                sw = min(len(completed_successes), 32)
                record["ep_success_mean"] = sum(completed_successes[-sw:]) / sw
            if completed_fails:
                fw = min(len(completed_fails), 32)
                record["ep_fail_mean"] = sum(completed_fails[-fw:]) / fw
                record["ep_fail_count"] = sum(1 for f in completed_fails if f > 0)
            if completed_failsA:
                faw = min(len(completed_failsA), 32)
                record["ep_failA_mean"] = sum(completed_failsA[-faw:]) / faw
                record["ep_failA_count"] = sum(1 for f in completed_failsA if f > 0)
            record["ep_success_count"] = sum(1 for s in completed_successes if s > 0)
            record["ep_count"] = len(completed_returns)
        logger.log(record)
        print(
            f"[iter {it:04d}] "
            f"shaped_r={rollout_r_info['mean_shaped_reward']:+.3f} "
            f"pg={metrics['ppo/pg_loss']:+.4f} "
            f"v={metrics['ppo/value_loss']:.3f} "
            f"ent={metrics['ppo/entropy']:.3f} "
            f"bc={metrics['ppo/bc_loss']:.3f} "
            f"kl={metrics['ppo/approx_kl']:.4f} "
            f"t={t_iter:.1f}s "
            + (f"ep_ret={record['ep_return_mean']:+.2f} "
               f"ep_len={record['ep_len_mean']:.0f} "
               f"ep_n={record['ep_count']} "
               f"succ={record.get('ep_success_count', 0)} "
               f"fail={record.get('ep_fail_count', 0)}" if "ep_return_mean" in record else ""),
            flush=True,
        )

        if (it + 1) % cfg.ckpt_every_iters == 0:
            _save_ckpt(policy, cfg.log_dir, it + 1)

    _save_ckpt(policy, cfg.log_dir, cfg.total_iters)
    try:
        env.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()
    cfg = _apply_overrides(Config(), args)

    if args.dryrun_env:
        _dryrun_env(cfg)
        return
    train(cfg)


if __name__ == "__main__":
    main()
