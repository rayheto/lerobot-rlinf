"""Tail an openpi train.log and mirror metrics to TensorBoard.

openpi's train.py logs only to wandb. This tailer follows the train log,
parses two kinds of lines:

  Step 100: grad_norm=1.1378, loss=0.0648, param_norm=1803.7704
  04:06:47.791 [I] Progress on: 53.0it/200it rate:1.6s/it remaining:... elapsed:...

and writes scalars under `train/*` to a TB events file. It also recomputes
the LR from openpi's warmup-cosine schedule using the same defaults the
training script uses (overridable via CLI flags).

Run it as a sidecar to the training process — no upstream changes required.

Example:
    python src/tb_tailer.py /tmp/sft_run/train.log /tmp/sft_run/tb
    tensorboard --logdir /tmp/sft_run/tb --port 6006
"""

from __future__ import annotations

import argparse
import datetime
import math
import re
import sys
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


STEP_RE = re.compile(r"Step (\d+):\s*(.+)$")
KV_RE = re.compile(r"(\w+)=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
# matches "Progress on: 53.0it/25.0kit rate:1.6s/it ..."
PROG_RE = re.compile(
    r"Progress on:\s+([\d.]+)(k?)it/[\d.]+(k?)it\s+rate:([\d.]+)s/it"
)


def cosine_lr(step: int, peak_lr: float, warmup_steps: int,
              decay_steps: int, decay_lr: float) -> float:
    """Match optax.warmup_cosine_decay_schedule used in openpi."""
    init_value = peak_lr / (warmup_steps + 1)
    if step < warmup_steps:
        # linear warmup from init to peak
        return init_value + (peak_lr - init_value) * (step / max(1, warmup_steps))
    # cosine decay from peak to decay_lr over decay_steps
    progress = (step - warmup_steps) / max(1, decay_steps)
    if progress >= 1.0:
        return decay_lr
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return decay_lr + (peak_lr - decay_lr) * cos


def _follow(path: Path):
    while not path.exists():
        time.sleep(1.0)
    with path.open("r") as f:
        f.seek(0, 2)
        while True:
            line = f.readline()
            if line:
                yield line
            else:
                time.sleep(0.5)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("log_path", type=Path)
    p.add_argument("tb_logdir", type=Path)
    p.add_argument("--from-start", action="store_true",
                   help="Replay from the start of the log instead of tailing.")
    # openpi CosineDecaySchedule defaults (third_party/openpi/.../optimizer.py)
    p.add_argument("--peak-lr", type=float, default=2.5e-5)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--decay-steps", type=int, default=30_000)
    p.add_argument("--decay-lr", type=float, default=2.5e-6)
    args = p.parse_args()

    # Each invocation gets its own timestamped subdir so TensorBoard shows a
    # distinct, identifiable run instead of collapsing every restart into the
    # same unnamed "." run under tb_logdir.
    run_dir = args.tb_logdir / datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))
    print(f"tb_tailer: {args.log_path} -> {run_dir}", flush=True)
    print(
        f"  lr schedule: peak={args.peak_lr} warmup={args.warmup_steps} "
        f"decay_steps={args.decay_steps} decay_lr={args.decay_lr}",
        flush=True,
    )

    last_prog_step: int | None = None

    def handle(line: str) -> None:
        nonlocal last_prog_step
        m = STEP_RE.search(line)
        if m:
            step = int(m.group(1))
            for k, v in KV_RE.findall(m.group(2)):
                writer.add_scalar(f"train/{k}", float(v), step)
            writer.add_scalar(
                "train/lr",
                cosine_lr(step, args.peak_lr, args.warmup_steps,
                          args.decay_steps, args.decay_lr),
                step,
            )
            writer.flush()
            print(f"step={step}", flush=True)
            return

        mp = PROG_RE.search(line)
        if mp:
            count = float(mp.group(1))
            if mp.group(2) == "k":
                count *= 1000
            step = int(count)
            rate = float(mp.group(4))
            # avoid duplicate writes if the same step is logged again
            if last_prog_step != step:
                writer.add_scalar("train/step_time_s", rate, step)
                writer.flush()
                last_prog_step = step

    try:
        if args.from_start and args.log_path.exists():
            for line in args.log_path.read_text().splitlines():
                handle(line)
        for line in _follow(args.log_path):
            handle(line)
    except KeyboardInterrupt:
        pass
    finally:
        writer.close()


if __name__ == "__main__":
    sys.exit(main())
