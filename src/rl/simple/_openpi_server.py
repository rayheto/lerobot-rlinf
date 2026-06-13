"""Shared helpers to spawn / wait / kill the openpi JAX serve_policy.py process.

Used by `train.py` and `eval.py`. Lives in its own module so callers can
import without circular dependency on `train.py`.
"""
from __future__ import annotations

import os
import pathlib
import shlex
import signal
import socket
import subprocess
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
OPENPI_ROOT = REPO_ROOT / "third_party" / "openpi"
OPENPI_PY = OPENPI_ROOT / ".venv" / "bin" / "python"
OPENPI_CLIENT_SRC = OPENPI_ROOT / "packages" / "openpi-client" / "src"


def ensure_openpi_client_on_path() -> None:
    """Add openpi-client (pure python) to sys.path. Idempotent."""
    if str(OPENPI_CLIENT_SRC) not in sys.path:
        sys.path.insert(0, str(OPENPI_CLIENT_SRC))


def spawn_openpi_server(
    ckpt: pathlib.Path, config_name: str, prompt: str, port: int
) -> subprocess.Popen:
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
    return subprocess.Popen(
        cmd, cwd=str(OPENPI_ROOT), env=env, start_new_session=True
    )


def wait_for_port(host: str, port: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            try:
                s.connect((host, port))
                return
            except OSError:
                time.sleep(2.0)
    sys.exit(f"openpi server on {host}:{port} did not come up within {timeout_s}s")


def kill_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        print(f"[openpi-server] stopping (pid={proc.pid})...", flush=True)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
