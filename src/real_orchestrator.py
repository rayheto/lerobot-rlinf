"""Orchestrator for real-robot policy evaluation.

Mirrors src/eval.py but targets real hardware instead of Isaac Lab:

  1. Spawns the OpenPI policy server (JAX venv, numpy>=2).
  2. Waits for the server to bind its port.
  3. Runs src/real.py eval in the current Python environment (no Isaac Sim).
  4. Kills the server on exit.

For teleoperation data collection (src/real.py collect), no policy server is
needed — run it directly.

Example:
    python src/real_orchestrator.py \\
        --exp-name=so101_pick_orange_lora_v0 \\
        --eval-rounds=5 \\
        --follower-port=/dev/ttyACM1
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
REAL_PY = REPO_ROOT / "src" / "real.py"


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
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return subprocess.Popen(cmd, cwd=str(OPENPI_ROOT), env=env, start_new_session=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Real-robot eval: spawn OpenPI server + run src/real.py eval.",
    )
    # Checkpoint args (same as eval.py)
    p.add_argument("--config-name", default="pi05_lora_so101_pick_orange")
    p.add_argument("--exp-name", help="Experiment name under outputs/<config>/.")
    p.add_argument("--checkpoint-base-dir", default=str(REPO_ROOT / "outputs"))
    p.add_argument("--checkpoint-dir", default=None,
                   help="Absolute path to a step ckpt dir; overrides --exp-name/--step.")
    p.add_argument("--step", type=int, default=None)

    # Server args
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--server-timeout-s", type=float, default=180.0)

    # Real-robot eval args (forwarded to src/real.py eval)
    p.add_argument("--prompt", default="Grab orange and place into plate")
    p.add_argument("--action-horizon", type=int, default=10)
    p.add_argument("--eval-rounds", type=int, default=20)
    p.add_argument("--episode-length-s", type=float, default=60.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--dataset-dir", default=None,
                   help="Output dir. Default: <ckpt>/real_dataset/.")
    p.add_argument("--follower-port", default="/dev/ttyACM1")
    p.add_argument("--follower-calib", default=".cache/so101_follower.json")
    p.add_argument("--front-camera-index", type=int, default=0)
    p.add_argument("--wrist-camera-index", type=int, default=2)
    p.add_argument("--camera-width", type=int, default=640)
    p.add_argument("--camera-height", type=int, default=480)
    p.add_argument("--front-camera-calib", default=None)
    p.add_argument("--wrist-camera-calib", default=None)
    p.add_argument("--recalibrate", action="store_true")

    args = p.parse_args()

    if not args.checkpoint_dir and not args.exp_name:
        sys.exit("must provide --exp-name or --checkpoint-dir")

    ckpt = _resolve_ckpt(args)
    dataset_dir = Path(args.dataset_dir).resolve() if args.dataset_dir else (ckpt / "real_dataset")
    print(f"checkpoint: {ckpt}", flush=True)
    print(f"dataset dir: {dataset_dir}", flush=True)

    server = _spawn_server(ckpt, args.config_name, args.prompt, args.port)
    rc = 1
    try:
        _wait_for_port(args.host, args.port, args.server_timeout_s)
        print(f"policy server ready on {args.host}:{args.port}", flush=True)

        cmd = [
            sys.executable, str(REAL_PY), "eval",
            "--policy_host", args.host,
            "--policy_port", str(args.port),
            "--policy_action_horizon", str(args.action_horizon),
            "--policy_language_instruction", args.prompt,
            "--eval_rounds", str(args.eval_rounds),
            "--episode_length_s", str(args.episode_length_s),
            "--fps", str(args.fps),
            "--dataset_dir", str(dataset_dir),
            "--follower_port", args.follower_port,
            "--follower_calib", args.follower_calib,
            "--front_camera_index", str(args.front_camera_index),
            "--wrist_camera_index", str(args.wrist_camera_index),
            "--camera_width", str(args.camera_width),
            "--camera_height", str(args.camera_height),
        ]
        if args.front_camera_calib:
            cmd += ["--front_camera_calib", args.front_camera_calib]
        if args.wrist_camera_calib:
            cmd += ["--wrist_camera_calib", args.wrist_camera_calib]
        if args.recalibrate:
            cmd += ["--recalibrate"]

        print(f"\n$ {shlex.join(cmd)}", flush=True)
        rc = subprocess.run(cmd).returncode
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
