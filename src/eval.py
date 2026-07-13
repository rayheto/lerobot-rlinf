"""End-to-end eval for SO-101 pick-orange in Isaac Lab.

Two-process orchestrator (single venv won't work — numpy version split):

  1. openpi serve_policy.py runs in third_party/openpi/.venv (JAX, numpy>=2)
     and exposes the trained LoRA checkpoint over WebSocket.
  2. src/eval_run.py runs in .venv-isaacsim (Isaac Sim 5.1 pins numpy==1.26),
     spawns the env headless, writes one mp4 per episode, and queries the
     openpi server with the next chunk prefetched concurrently while the
     current chunk plays.

See docs/notes.md "Install fiddles" for the numpy split rationale.

Checkpoints land at outputs/<config>/<exp>/<step>/ (see sft_train.py); we
auto-pick the latest step subdir under --exp-name if --step is omitted.

Example:
    python src/eval.py --exp-name=so101_pick_orange_lora_v0 --eval-rounds=20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_ROOT = REPO_ROOT / "third_party" / "openpi"
OPENPI_PY = OPENPI_ROOT / ".venv" / "bin" / "python"
ISAAC_PY = REPO_ROOT / ".venv-isaacsim" / "bin" / "python"
LEISAAC_ROOT = REPO_ROOT / "third_party" / "leisaac"
# We replaced leisaac's policy_inference.py with our own loop (headless +
# video capture + async chunk prefetch). See src/eval_run.py.
EVAL_RUN = REPO_ROOT / "src" / "eval_run.py"
# leisaac's constant.py picks the outer git root as ASSETS_ROOT, which
# resolves to <repo>/assets in our nested layout — wrong. Pin it explicitly.
LEISAAC_ASSETS = LEISAAC_ROOT / "assets"

DEFAULT_CONFIG = "pi05_lora_so101_pick_orange"
DEFAULT_TASK = "LeIsaac-SO101-PickOrange-v0"
# Match the training dataset: all 60 demos in EverNorif/leisaac-pick-orange
# share this single-orange instruction. The LeIsaac env still spawns 3
# oranges and only flags success when ALL three are plated + arm at rest,
# so the success ceiling here is capped — useful for behavior-checking
# (does the arm grasp + place cleanly?), not full task completion.
DEFAULT_PROMPT = "Grab orange and place into plate"
# pi05_lora_so101_pick_orange uses action_horizon=10 (see openpi config.py:957).
DEFAULT_ACTION_HORIZON = 10
# Training demos were recorded at 30 fps. Stepping the env at 60 Hz makes
# each chunked trajectory play 2x faster than what the model saw at train
# time — causes jerky tracking. Match it.
DEFAULT_STEP_HZ = 30
DEFAULT_BACKEND = "sft"
REAL_BACKEND = REPO_ROOT / "src" / "real_backend.py"
DEFAULT_BASE_CKPT = (
    REPO_ROOT
    / "outputs"
    / DEFAULT_CONFIG
    / "so101_pick_orange_lora_v0"
    / "4999"
)


def _pick_latest_step(exp_dir: Path) -> int:
    steps = [int(p.name) for p in exp_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not steps:
        sys.exit(f"no step subdirs under {exp_dir}")
    return max(steps)


def _find_checkpoint_steps(exp_dir: Path) -> list[int]:
    steps = []
    for p in exp_dir.iterdir():
        if p.is_dir() and p.name.isdigit() and (p / "_CHECKPOINT_METADATA").exists():
            steps.append(int(p.name))
    return sorted(steps)


def _resolve_ckpt(args: argparse.Namespace) -> Path:
    if args.checkpoint_dir:
        ckpt = Path(args.checkpoint_dir).resolve()
        if not ckpt.is_dir():
            sys.exit(f"checkpoint dir does not exist: {ckpt}")
        return ckpt
    exp_dir = Path(args.checkpoint_base_dir).resolve() / args.config_name / args.exp_name
    if not exp_dir.is_dir():
        sys.exit(f"experiment dir does not exist: {exp_dir}")
    step = args.step if args.step is not None else _pick_latest_step(exp_dir)
    ckpt = exp_dir / str(step)
    if not ckpt.is_dir():
        sys.exit(f"checkpoint step dir does not exist: {ckpt}")
    return ckpt


def _wait_for_port(host: str, port: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            try:
                s.connect((host, port))
                return
            except OSError:
                time.sleep(1.0)
    sys.exit(f"policy server on {host}:{port} did not come up within {timeout_s}s")


def _port_is_free(host: str, port: int) -> bool:
    bind_host = "127.0.0.1" if host in {"localhost", "0.0.0.0"} else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((bind_host, port))
        except OSError:
            return False
    return True


def _spawn_server(
    ckpt: Path,
    config_name: str,
    prompt: str,
    port: int,
    xla_mem_fraction: float,
) -> subprocess.Popen:
    if not OPENPI_PY.exists():
        sys.exit(f"openpi venv missing at {OPENPI_PY}")
    # tyro union-of-dataclasses syntax: `policy:checkpoint` selects the
    # Checkpoint variant; --policy.config / --policy.dir set its fields.
    cmd = [
        str(OPENPI_PY),
        str(OPENPI_ROOT / "scripts" / "serve_policy.py"),
        "--port", str(port),
        "--default_prompt", prompt,
        "policy:checkpoint",
        "--policy.config", config_name,
        "--policy.dir", str(ckpt),
    ]
    print(f"\n$ (cwd={OPENPI_ROOT}) {shlex.join(cmd)}", flush=True)
    env = os.environ.copy()
    # Training used 0.9 to fit bs=8 LoRA; inference only needs the model
    # weights (~6 GB bf16). Disable preallocation so Vulkan/IsaacSim can
    # share the GPU on the same card.
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(xla_mem_fraction))
    env.setdefault("PYTHONUNBUFFERED", "1")
    # New session so we can kill the whole tree (server may spawn JAX workers).
    return subprocess.Popen(cmd, cwd=str(OPENPI_ROOT), env=env, start_new_session=True)


def _spawn_residual_server(
    args: argparse.Namespace,
    rl_ckpt: Path,
    base_port: int,
    port: int,
    isaac_py: Path,
) -> subprocess.Popen:
    cmd = [
        str(isaac_py),
        "-u",
        "-m",
        "src.rl.rlinf_residual.eval_server",
        "--rl-checkpoint-dir",
        str(rl_ckpt),
        "--base-host",
        args.host,
        "--base-port",
        str(base_port),
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--obs-dim",
        str(args.rlinf_obs_dim),
        "--action-dim",
        "6",
        "--residual-clip",
        str(args.residual_clip),
        "--device",
        args.rlinf_device,
    ]
    print(f"\n$ (cwd={REPO_ROOT}) {shlex.join(cmd)}", flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    return subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, start_new_session=True)


def _run_leisaac(args: argparse.Namespace, isaac_py: Path, port: int, dataset_dir: Path) -> int:
    if not isaac_py.exists():
        sys.exit(f"isaac venv python missing at {isaac_py}")
    if not EVAL_RUN.exists():
        sys.exit(f"eval_run script missing at {EVAL_RUN}")
    cmd = [
        str(isaac_py), "-u", str(EVAL_RUN),
        "--task", args.task,
        "--policy_host", args.host,
        "--policy_port", str(port),
        "--policy_action_horizon", str(args.action_horizon),
        "--policy_language_instruction", args.prompt,
        "--eval_rounds", str(args.eval_rounds),
        "--episode_length_s", str(args.episode_length_s),
        "--step_hz", str(args.step_hz),
        "--dataset_dir", str(dataset_dir),
        "--dataset_fps", str(args.dataset_fps),
        "--prefetch-latency-ms", str(args.prefetch_latency_ms),
    ]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if not args.prefetch:
        cmd += ["--no-prefetch"]
    env = os.environ.copy()
    env["LEISAAC_ASSETS_ROOT"] = str(LEISAAC_ASSETS)
    print(f"\n$ LEISAAC_ASSETS_ROOT={LEISAAC_ASSETS} {shlex.join(cmd)}", flush=True)
    return subprocess.run(cmd, env=env).returncode


def _run_real_backend(args: argparse.Namespace, port: int, ckpt: Path | None = None) -> int:
    if not REAL_BACKEND.exists():
        sys.exit(f"real backend script missing at {REAL_BACKEND}")
    if not OPENPI_PY.exists():
        sys.exit(f"openpi venv missing at {OPENPI_PY}")
    if not args.real_config:
        sys.exit("--backend=real requires --real-config")

    cmd = [
        str(OPENPI_PY), "-u", str(REAL_BACKEND),
        "--config", args.real_config,
        "--policy-host", args.host,
        "--policy-port", str(port),
        "--prompt", args.prompt,
        "--action-horizon", str(args.action_horizon),
        "--step-hz", str(args.step_hz),
    ]
    if args.real_max_steps is not None:
        cmd += ["--max-steps", str(args.real_max_steps)]
    debug_dir = args.real_debug_dir
    if debug_dir is None and ckpt is not None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        debug_dir = str(ckpt / "real_debug" / stamp)
    if debug_dir:
        cmd += ["--debug-dir", debug_dir]
    if args.real_dry_run:
        cmd.append("--dry-run")
    if args.real_detect_cameras:
        cmd.append("--detect-cameras")

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    print(f"\n$ (cwd={REPO_ROOT}) {shlex.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=str(REPO_ROOT), env=env).returncode


def _load_state(state_path: Path) -> dict:
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {}


def _save_state(state_path: Path, state: dict) -> None:
    state_path.write_text(json.dumps(state, indent=2))


def _query_gpus() -> list[dict]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        sys.exit(f"failed to query GPUs with nvidia-smi: {exc}")

    gpus: list[dict] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            sys.exit(f"unexpected nvidia-smi output line: {line}")
        idx, name, mem_used, mem_total, util = parts
        gpus.append({
            "index": int(idx),
            "name": name,
            "memory_used_mib": int(mem_used),
            "memory_total_mib": int(mem_total),
            "utilization_gpu_pct": int(util),
        })
    return gpus


def _is_idle_gpu(gpu: dict, max_mem_mib: int, max_util_pct: int) -> bool:
    return (
        gpu["memory_used_mib"] <= max_mem_mib
        and gpu["utilization_gpu_pct"] <= max_util_pct
    )


def _select_gpus(args: argparse.Namespace) -> list[int]:
    all_gpus = _query_gpus()
    by_index = {g["index"]: g for g in all_gpus}
    idle = [
        g for g in all_gpus
        if _is_idle_gpu(g, args.gpu_memory_used_max_mib, args.gpu_util_max_pct)
    ]

    if args.gpus == "auto":
        selected = [g["index"] for g in idle]
        if args.max_gpus > 0:
            selected = selected[:args.max_gpus]
    else:
        try:
            selected = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
        except ValueError:
            sys.exit(f"invalid --gpus value: {args.gpus}")
        missing = [g for g in selected if g not in by_index]
        if missing:
            sys.exit(f"requested GPU(s) not found: {missing}")
        busy = [
            by_index[g] for g in selected
            if not _is_idle_gpu(by_index[g], args.gpu_memory_used_max_mib, args.gpu_util_max_pct)
        ]
        if busy:
            details = ", ".join(
                f"{g['index']} mem={g['memory_used_mib']}MiB util={g['utilization_gpu_pct']}%"
                for g in busy
            )
            sys.exit(
                "requested GPU(s) are not idle under the configured thresholds: "
                f"{details}"
            )

    if len(selected) < args.min_gpus:
        snapshot = ", ".join(
            f"{g['index']} mem={g['memory_used_mib']}MiB util={g['utilization_gpu_pct']}%"
            for g in all_gpus
        )
        sys.exit(
            f"only {len(selected)} idle GPU(s) found, need at least {args.min_gpus}. "
            f"Snapshot: {snapshot}"
        )
    return selected


def _check_shard_ports(host: str, base_port: int, n: int) -> None:
    busy = [p for p in range(base_port, base_port + n) if not _port_is_free(host, p)]
    if busy:
        sys.exit(f"port(s) already in use: {busy}. Pick another --base-port.")


def _kill_eval_proc(proc: subprocess.Popen, port: int) -> None:
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=15)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    # If eval.py is killed before its own finally block runs, serve_policy.py
    # can remain in a separate process group and keep the shard port bound.
    subprocess.run(["pkill", "-9", "-f", f"serve_policy.py --port {port}"], check=False)


def _run_shard(
    args: argparse.Namespace,
    ckpt_dir: Path,
    gpu: int,
    shard_idx: int,
    step: int,
) -> subprocess.Popen:
    port = args.base_port + shard_idx
    dataset_dir = ckpt_dir / "eval" / f"shard_{shard_idx}"
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--backend", args.backend,
        "--config-name", args.config_name,
        "--checkpoint-dir", str(ckpt_dir),
        "--task", args.task,
        "--prompt", args.prompt,
        "--action-horizon", str(args.action_horizon),
        "--eval-rounds", str(args.rounds_per_shard),
        "--episode-length-s", str(args.episode_length_s),
        "--step-hz", str(args.step_hz),
        "--host", args.host,
        "--port", str(port),
        "--server-timeout-s", str(args.server_timeout_s),
        "--dataset-dir", str(dataset_dir),
        "--dataset-fps", str(args.dataset_fps),
        "--prefetch-latency-ms", str(args.prefetch_latency_ms),
        "--seed", str(step * 100 + shard_idx),
    ]
    if args.isaac_python:
        cmd += ["--isaac-python", args.isaac_python]
    if args.prefetch:
        cmd.append("--prefetch")
    else:
        cmd.append("--no-prefetch")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    print(
        f"[step {step}] shard {shard_idx}: gpu={gpu} port={port} -> {dataset_dir}",
        flush=True,
    )
    return subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, start_new_session=True)


def _aggregate(shard_dirs: list[Path], dataset_fps: int, fast_threshold_s: float) -> dict:
    fast_threshold_frames = fast_threshold_s * dataset_fps
    outcomes: list[bool] = []
    lengths: list[int | None] = []

    for d in shard_dirs:
        results_path = d / "results.json"
        if not results_path.exists():
            print(f"eval: missing {results_path}, shard likely failed; excluded", flush=True)
            continue
        results = json.loads(results_path.read_text())

        ep_lengths: dict[int, int] = {}
        episodes_path = d / "meta" / "episodes.jsonl"
        if episodes_path.exists():
            for line in episodes_path.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                ep_lengths[rec["episode_index"]] = rec["length"]

        for i, success in enumerate(results.get("outcomes", [])):
            outcomes.append(bool(success))
            lengths.append(ep_lengths.get(i))

    n = len(outcomes)
    n_success = sum(outcomes)
    success_rate = n_success / n if n else 0.0
    success_std = math.sqrt(success_rate * (1 - success_rate) / n) if n else 0.0
    n_fast = sum(
        1 for success, length in zip(outcomes, lengths)
        if success and length is not None and length <= fast_threshold_frames
    )
    fast_rate = n_fast / n if n else 0.0
    return {
        "n_episodes": n,
        "n_success": n_success,
        "success_rate": success_rate,
        "success_std": success_std,
        "n_fast": n_fast,
        "fast_rate": fast_rate,
    }


def _run_checkpoint_watch_eval(
    args: argparse.Namespace,
    ckpt_dir: Path,
    step: int,
    gpus: list[int],
) -> dict:
    _check_shard_ports(args.host, args.base_port, len(gpus))
    procs = [_run_shard(args, ckpt_dir, gpu, i, step) for i, gpu in enumerate(gpus)]
    deadline = time.time() + args.shard_timeout_s
    for i, proc in enumerate(procs):
        remaining = max(0.0, deadline - time.time())
        port = args.base_port + i
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            print(f"[step {step}] shard {i} timed out after {args.shard_timeout_s:.0f}s", flush=True)
            _kill_eval_proc(proc, port)
        else:
            if proc.returncode != 0:
                print(f"[step {step}] shard {i} exited with code {proc.returncode}", flush=True)

    shard_dirs = [ckpt_dir / "eval" / f"shard_{i}" for i in range(len(gpus))]
    summary = _aggregate(shard_dirs, args.dataset_fps, args.fast_threshold_s)
    summary.update({
        "backend": args.backend,
        "config_name": args.config_name,
        "exp_name": args.exp_name,
        "checkpoint_dir": str(ckpt_dir),
        "step": step,
        "gpus": gpus,
        "base_port": args.base_port,
    })
    return summary


def run_watch(args: argparse.Namespace) -> None:
    if args.backend != "sft":
        sys.exit(f"--watch currently supports only --backend=sft, got {args.backend}")
    if not args.exp_name:
        sys.exit("--watch requires --exp-name")

    exp_dir = Path(args.checkpoint_base_dir).resolve() / args.config_name / args.exp_name
    if not exp_dir.is_dir():
        sys.exit(f"experiment dir does not exist: {exp_dir}")

    gpus = _select_gpus(args)
    state_path = exp_dir / "eval_watch_state.json"
    summary_path = exp_dir / "eval_summary.jsonl"
    state = _load_state(state_path)
    seen_steps: set[int] = {int(k) for k in state}

    print(
        f"eval: watching {exp_dir} backend={args.backend} gpus={gpus} "
        f"poll={args.poll_interval_s}s already_evaluated={sorted(seen_steps) or 'none'}",
        flush=True,
    )

    while True:
        available = _find_checkpoint_steps(exp_dir)
        for step in [s for s in available if s not in seen_steps]:
            ckpt_dir = exp_dir / str(step)
            print(
                f"[step {step}] checkpoint detected, starting eval "
                f"({len(gpus)} shards x {args.rounds_per_shard} rounds)",
                flush=True,
            )
            t0 = time.time()
            summary = _run_checkpoint_watch_eval(args, ckpt_dir, step, gpus)
            summary["wall_time_s"] = time.time() - t0
            (ckpt_dir / "eval").mkdir(parents=True, exist_ok=True)
            (ckpt_dir / "eval" / "summary.json").write_text(json.dumps(summary, indent=2))
            with summary_path.open("a") as f:
                f.write(json.dumps(summary) + "\n")
            print(
                f"[step {step}] success_rate={summary['success_rate']:.3f}"
                f"+/-{summary['success_std']:.3f} ({summary['n_success']}/{summary['n_episodes']})"
                f"  fast_rate={summary['fast_rate']:.3f} ({summary['n_fast']}/{summary['n_episodes']})"
                f"  wall={summary['wall_time_s']:.0f}s",
                flush=True,
            )
            seen_steps.add(step)
            state[str(step)] = summary
            _save_state(state_path, state)

        if args.until_step is not None:
            if args.until_step in seen_steps:
                print(f"eval: reached --until-step={args.until_step}, exiting", flush=True)
                return
            if any(s > args.until_step for s in available) and args.until_step not in available:
                print(
                    f"eval: --until-step={args.until_step} was evicted before evaluation; exiting",
                    flush=True,
                )
                return

        time.sleep(args.poll_interval_s)


def run_once(args: argparse.Namespace) -> int:
    if args.backend not in {"sft", "rlinf-residual", "real"}:
        sys.exit(f"unsupported --backend={args.backend}")
    if args.backend == "real" and args.real_detect_cameras:
        return _run_real_backend(args, args.port)
    if not args.checkpoint_dir and not args.exp_name:
        sys.exit("must provide --exp-name or --checkpoint-dir")

    ckpt = _resolve_ckpt(args)
    isaac_py = Path(args.isaac_python) if args.isaac_python else ISAAC_PY
    dataset_dir = Path(args.dataset_dir).resolve() if args.dataset_dir else (ckpt / "dataset")
    print(f"checkpoint: {ckpt}", flush=True)
    print(f"isaac python: {isaac_py}", flush=True)
    print(f"dataset dir: {dataset_dir}", flush=True)
    xla_mem_fraction = (
        args.xla_mem_fraction
        if args.xla_mem_fraction is not None
        else (0.85 if args.backend == "real" else 0.35)
    )

    procs: list[tuple[subprocess.Popen, int | None]] = []
    rc = 1
    try:
        if args.backend in {"sft", "real"}:
            server = _spawn_server(
                ckpt,
                args.config_name,
                args.prompt,
                args.port,
                xla_mem_fraction,
            )
            procs.append((server, args.port))
        else:
            base_ckpt = Path(args.base_checkpoint_dir).resolve()
            if not base_ckpt.is_dir():
                sys.exit(f"base checkpoint dir does not exist: {base_ckpt}")
            base_port = args.base_policy_port or (args.port + 1)
            base_server = _spawn_server(
                base_ckpt,
                args.base_config_name,
                args.prompt,
                base_port,
                xla_mem_fraction,
            )
            procs.append((base_server, base_port))
            _wait_for_port(args.host, base_port, args.server_timeout_s)
            residual_server = _spawn_residual_server(
                args, ckpt, base_port, args.port, isaac_py
            )
            procs.append((residual_server, args.port))

        _wait_for_port(args.host, args.port, args.server_timeout_s)
        print(f"policy server ready on {args.host}:{args.port}", flush=True)
        if args.backend == "real":
            rc = _run_real_backend(args, args.port, ckpt)
        else:
            rc = _run_leisaac(args, isaac_py, args.port, dataset_dir)
    finally:
        for proc, port in reversed(procs):
            if proc.poll() is not None:
                continue
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if port is not None:
                subprocess.run(
                    ["pkill", "-9", "-f", f"serve_policy.py --port {port}"],
                    check=False,
                )
    if args.backend == "real":
        summary = {
            "n_episodes": 0,
            "n_success": 0,
            "success_rate": 0.0,
            "success_std": 0.0,
            "n_fast": 0,
            "fast_rate": 0.0,
        }
    else:
        summary = _aggregate([dataset_dir], args.dataset_fps, args.fast_threshold_s)
    summary.update(
        {
            "backend": args.backend,
            "config_name": args.config_name,
            "checkpoint_dir": str(ckpt),
            "dataset_dir": str(dataset_dir),
            "returncode": rc,
        }
    )
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"eval summary: {dataset_dir / 'summary.json'}", flush=True)
    return rc


def main() -> None:
    p = argparse.ArgumentParser(description="SO-101 pick-orange eval (openpi server + leisaac client).")
    p.add_argument("--backend", default=DEFAULT_BACKEND, choices=["sft", "rlinf-residual", "real"],
                   help="Eval backend.")
    p.add_argument("--watch", action="store_true",
                   help="Watch an experiment directory and evaluate each new checkpoint.")
    p.add_argument("--config-name", default=DEFAULT_CONFIG)
    p.add_argument("--exp-name", help="Experiment name under outputs/<config>/.")
    p.add_argument("--checkpoint-base-dir", default=str(REPO_ROOT / "outputs"))
    p.add_argument("--checkpoint-dir", default=None,
                   help="Absolute path to a step ckpt dir; overrides --exp-name/--step.")
    p.add_argument("--base-config-name", default=DEFAULT_CONFIG,
                   help="Base pi05 config for --backend=rlinf-residual.")
    p.add_argument("--base-checkpoint-dir", default=str(DEFAULT_BASE_CKPT),
                   help="Frozen pi05 checkpoint dir for --backend=rlinf-residual.")
    p.add_argument("--base-policy-port", type=int, default=None,
                   help="Base pi05 server port for --backend=rlinf-residual. Default: --port+1.")
    p.add_argument("--rlinf-obs-dim", type=int, default=12)
    p.add_argument("--rlinf-device", default="cpu")
    p.add_argument("--residual-clip", type=float, default=0.0)
    p.add_argument("--step", type=int, default=None,
                   help="Checkpoint step to load. Default: latest under exp dir.")
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--action-horizon", type=int, default=DEFAULT_ACTION_HORIZON)
    p.add_argument("--eval-rounds", type=int, default=20)
    p.add_argument("--episode-length-s", type=float, default=60.0)
    p.add_argument("--step-hz", type=int, default=DEFAULT_STEP_HZ)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--server-timeout-s", type=float, default=180.0,
                   help="Max wait for openpi server to bind the port (JIT + weight load).")
    p.add_argument("--xla-mem-fraction", type=float, default=None,
                   help="XLA_PYTHON_CLIENT_MEM_FRACTION for the openpi policy server. "
                        "Default: 0.85 for --backend=real, otherwise 0.35.")
    p.add_argument("--isaac-python", default=None,
                   help="Override path to .venv-isaacsim python.")
    p.add_argument("--dataset-dir", default=None,
                   help="Root for the lerobot v2.1 output dataset (data/, videos/, meta/). "
                        "Default: <ckpt>/dataset/.")
    p.add_argument("--dataset-fps", type=int, default=30,
                   help="Output dataset fps. Must divide the env sim_fps (60 Hz) evenly. "
                        "Default 30 matches EverNorif/leisaac-pick-orange.")
    # Prefetch is OFF by default: the action chunk is computed from an obs
    # captured ~7 steps before the chunk is applied, so the policy plans from a
    # state that's behind where the arm actually ends up — manifests as a
    # visible jerk at every chunk boundary and dropped success rate. Enable
    # only after implementing temporal ensembling / receding-horizon execution.
    p.add_argument("--prefetch", action=argparse.BooleanOptionalAction, default=False,
                   help="Async-prefetch the next action chunk during current chunk playback. "
                        "Hides infer latency but degrades the policy due to stale obs.")
    p.add_argument("--prefetch-latency-ms", type=float, default=120.0,
                   help="Slack reserved for in-flight infer; fire next chunk this many ms before "
                        "current chunk ends. Set above measured infer latency (~103 ms).")
    p.add_argument("--real-config", default=None,
                   help="YAML/JSON config for --backend=real; declares SO-101 port and front/wrist cameras.")
    p.add_argument("--real-max-steps", type=int, default=None,
                   help="Real backend action-step limit. Default from config, or unlimited.")
    p.add_argument("--real-dry-run", action="store_true",
                   help="Run live camera/policy loop but print actions instead of sending them to the arm.")
    p.add_argument("--real-debug-dir", default=None,
                   help="Directory for real-backend policy input images/videos. "
                        "Default: <checkpoint>/real_debug/<timestamp>/.")
    p.add_argument("--real-detect-cameras", action="store_true",
                   help="Print detected OpenCV cameras before running the real backend.")
    p.add_argument("--gpus", default="auto",
                   help="Watch mode only: 'auto' or comma-separated GPU ids. Default: auto.")
    p.add_argument("--gpu-memory-used-max-mib", type=int, default=1024,
                   help="Watch mode idle-GPU threshold for memory.used.")
    p.add_argument("--gpu-util-max-pct", type=int, default=10,
                   help="Watch mode idle-GPU threshold for utilization.gpu.")
    p.add_argument("--min-gpus", type=int, default=1,
                   help="Watch mode requires at least this many idle GPUs.")
    p.add_argument("--max-gpus", type=int, default=0,
                   help="Watch mode auto-select cap. 0 means all idle GPUs.")
    p.add_argument("--base-port", type=int, default=8100,
                   help="Watch mode first policy-server port; shard i uses base-port+i.")
    p.add_argument("--rounds-per-shard", type=int, default=15)
    p.add_argument("--fast-threshold-s", type=float, default=30.0,
                   help="Episodes that succeed within this many seconds count as FAST.")
    p.add_argument("--poll-interval-s", type=float, default=15.0)
    p.add_argument("--shard-timeout-s", type=float, default=1800.0)
    p.add_argument("--until-step", type=int, default=None,
                   help="Watch mode exits once this step has been evaluated or missed.")
    args = p.parse_args()

    if args.watch:
        run_watch(args)
        return
    sys.exit(run_once(args))


if __name__ == "__main__":
    main()
