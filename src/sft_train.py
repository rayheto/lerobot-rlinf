"""
SFT entry point for pick-orange on SO-101 via openpi LoRA.

Thin orchestrator: openpi already has a full training loop in
third_party/openpi/scripts/train.py — we shell out to it instead of
re-implementing it.

The openpi training environment lives in third_party/openpi/.venv/ (built
via `uv sync` inside third_party/openpi/). This script invokes that venv's
python directly, so it works from ANY shell — no need to activate the
openpi venv first.

One-time setup:
    cd third_party/openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync

Example:
    # Source HF_TOKEN if dataset is gated or needs HF auth.
    source .env

    python src/sft_train.py --exp-name=so101_pick_orange_lora_v0

Default config (`pi05_lora_so101_pick_orange`) lives in EverNorif's
openpi fork at third_party/openpi/src/openpi/training/config.py and
trains LoRA π₀.₅ on the LightwheelAI/leisaac-pick-orange dataset (60 ep)
with batch_size=8 (fits 24GB).

Run order:
    1) compute_norm_stats.py for the config (writes
       assets/<config>/EverNorif/leisaac-pick-orange/norm_stats.json).
    2) train.py — auto-downloads pi05_base ckpt from gs://openpi-assets
       on first invocation (~5-10 GB to ~/.cache/openpi).
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_ROOT = REPO_ROOT / "third_party" / "openpi"
OPENPI_PY = OPENPI_ROOT / ".venv" / "bin" / "python"


def _check_env() -> None:
    if not (OPENPI_ROOT / "scripts" / "train.py").exists():
        sys.exit(
            f"openpi submodule missing at {OPENPI_ROOT}.\n"
            "  git submodule update --init --recursive"
        )
    if not OPENPI_PY.exists():
        sys.exit(
            f"openpi venv missing at {OPENPI_PY}.\n"
            f"  cd {OPENPI_ROOT} && GIT_LFS_SKIP_SMUDGE=1 uv sync"
        )


def _run(cmd: list[str], env: dict) -> None:
    # cwd=OPENPI_ROOT so the config's relative ./assets and ./checkpoints
    # resolve under third_party/openpi/.
    print(f"\n$ (cwd={OPENPI_ROOT}) {shlex.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=str(OPENPI_ROOT))


def compute_norm_stats(config_name: str, env: dict) -> None:
    script = OPENPI_ROOT / "scripts" / "compute_norm_stats.py"
    _run([str(OPENPI_PY), str(script), "--config-name", config_name], env=env)


def train(args: argparse.Namespace, env: dict) -> None:
    cmd: list[str] = [
        str(OPENPI_PY),
        str(OPENPI_ROOT / "scripts" / "train.py"),
        args.config_name,
        f"--exp-name={args.exp_name}",
    ]
    if args.num_train_steps is not None:
        cmd.append(f"--num-train-steps={args.num_train_steps}")
    if args.batch_size is not None:
        cmd.append(f"--batch-size={args.batch_size}")
    if args.log_interval is not None:
        cmd.append(f"--log-interval={args.log_interval}")
    if args.save_interval is not None:
        cmd.append(f"--save-interval={args.save_interval}")
    if args.checkpoint_base_dir is not None:
        # openpi resolves checkpoint_base_dir against cwd; use absolute so
        # checkpoints land where the user expects, not under openpi/.
        cmd.append(f"--checkpoint-base-dir={Path(args.checkpoint_base_dir).resolve()}")
    if args.resume:
        cmd.append("--resume")
    if args.overwrite:
        cmd.append("--overwrite")
    if not args.wandb:
        cmd.append("--no-wandb-enabled")
    # argparse REMAINDER keeps the leading "--" token, which tyro rejects;
    # strip it so users can write `-- --foo` ergonomically.
    extra = [a for a in args.extra if a != "--"]
    cmd.extend(extra)
    _run(cmd, env=env)


def main() -> None:
    p = argparse.ArgumentParser(description="openpi LoRA SFT for SO-101 pick-orange.")
    p.add_argument(
        "--config-name",
        default="pi05_lora_so101_pick_orange",
        help="openpi TrainConfig name (defined in "
        "third_party/openpi/src/openpi/training/config.py).",
    )
    p.add_argument("--exp-name", required=True, help="Experiment name (checkpoint dir).")
    p.add_argument("--num-train-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--log-interval", type=int, default=None,
                   help="Logging cadence in steps (openpi default 100).")
    p.add_argument("--save-interval", type=int, default=None,
                   help="Checkpoint cadence in steps (config default).")
    p.add_argument("--checkpoint-base-dir", default=str(REPO_ROOT / "outputs"),
                   help="Root for checkpoints. Final dir is "
                        "<root>/<config_name>/<exp_name>/. Default: outputs/.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable W&B logging (requires WANDB_API_KEY). Off by default.",
    )
    p.add_argument(
        "--skip-norm-stats",
        action="store_true",
        help="Skip compute_norm_stats (use once stats are on disk).",
    )
    p.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Trailing args forwarded to train.py (e.g. -- --data.repo_id=...).",
    )
    args = p.parse_args()

    _check_env()
    env = os.environ.copy()
    # JAX defaults to 75% GPU preallocation; LoRA π₀.₅ at bs=8 needs ~22.5 GB
    # on a 24 GB card, so 90% is required (per openpi README).
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    # pbar.write goes to sys.stdout, which is block-buffered when redirected
    # to a file → "Step N:" lines stall in buffer for ~70 lines. Force flush.
    env.setdefault("PYTHONUNBUFFERED", "1")

    if not args.skip_norm_stats:
        compute_norm_stats(args.config_name, env)
    train(args, env)


if __name__ == "__main__":
    main()
