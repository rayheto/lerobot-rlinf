"""End-to-end eval for SO-101 pick-orange in Isaac Lab.

Two-process orchestrator (single venv won't work — numpy version split):

  1. openpi serve_policy.py runs in third_party/openpi/.venv (JAX, numpy>=2)
     and exposes the trained LoRA checkpoint over WebSocket.
  2. src/eval_run.py runs in rlinf-isaacsim-env (Isaac Sim 5.1 pins
     numpy==1.26), spawns the env headless, writes one mp4 per episode,
     and queries the openpi server with the next chunk prefetched
     concurrently while the current chunk plays.

See docs/notes.md "Install fiddles" for the numpy split rationale.

Checkpoints land at outputs/<config>/<exp>/<step>/ (see sft_train.py); we
auto-pick the latest step subdir under --exp-name if --step is omitted.

Example:
    python src/eval.py --exp-name=so101_pick_orange_lora_v0 --eval-rounds=20
"""

from __future__ import annotations

import argparse
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


def _default_isaac_python() -> Path:
    """rlinf-isaacsim-env is the conda env with Isaac Sim 5.1 + leisaac."""
    base = os.environ.get("CONDA_PREFIX_1") or os.environ.get("CONDA_PREFIX") or "/home/hlei/miniconda3"
    # Walk up to the conda root if we're inside an env.
    root = Path(base)
    if root.name != "miniconda3" and (root.parent / "envs").is_dir():
        root = root.parent.parent
    return root / "envs" / "rlinf-isaacsim-env" / "bin" / "python"


def _pick_latest_step(exp_dir: Path) -> int:
    steps = [int(p.name) for p in exp_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not steps:
        sys.exit(f"no step subdirs under {exp_dir}")
    return max(steps)


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


def _spawn_server(ckpt: Path, config_name: str, prompt: str, port: int) -> subprocess.Popen:
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
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")
    env.setdefault("PYTHONUNBUFFERED", "1")
    # New session so we can kill the whole tree (server may spawn JAX workers).
    return subprocess.Popen(cmd, cwd=str(OPENPI_ROOT), env=env, start_new_session=True)


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


def main() -> None:
    p = argparse.ArgumentParser(description="SO-101 pick-orange eval (openpi server + leisaac client).")
    p.add_argument("--config-name", default=DEFAULT_CONFIG)
    p.add_argument("--exp-name", help="Experiment name under outputs/<config>/.")
    p.add_argument("--checkpoint-base-dir", default=str(REPO_ROOT / "outputs"))
    p.add_argument("--checkpoint-dir", default=None,
                   help="Absolute path to a step ckpt dir; overrides --exp-name/--step.")
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
    p.add_argument("--isaac-python", default=None,
                   help="Override path to rlinf-isaacsim-env python.")
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
    args = p.parse_args()

    if not args.checkpoint_dir and not args.exp_name:
        sys.exit("must provide --exp-name or --checkpoint-dir")

    ckpt = _resolve_ckpt(args)
    isaac_py = Path(args.isaac_python) if args.isaac_python else _default_isaac_python()
    dataset_dir = Path(args.dataset_dir).resolve() if args.dataset_dir else (ckpt / "dataset")
    print(f"checkpoint: {ckpt}", flush=True)
    print(f"isaac python: {isaac_py}", flush=True)
    print(f"dataset dir: {dataset_dir}", flush=True)

    server = _spawn_server(ckpt, args.config_name, args.prompt, args.port)
    rc = 1
    try:
        _wait_for_port(args.host, args.port, args.server_timeout_s)
        print(f"policy server ready on {args.host}:{args.port}", flush=True)
        rc = _run_leisaac(args, isaac_py, args.port, dataset_dir)
    finally:
        if server.poll() is None:
            try:
                os.killpg(server.pid, signal.SIGTERM)
                server.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(server.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    sys.exit(rc)


if __name__ == "__main__":
    main()
