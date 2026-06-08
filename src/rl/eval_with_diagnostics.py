"""RL eval → diagnostics → tensorboard bridge.

Given an RL eval rollout that has been written to a LeRobot v2.1 dataset
(by the existing ``src/eval.py`` two-process orchestrator or any equivalent
producer), this script:

1. Runs the diagnostic modules listed in
   ``src.diagnostics.online_callback.DEFAULT_MODULES`` against
   ``(--ref-dataset, --cand-dataset)``.
2. Writes ratios as scalars into the TensorBoard event file under
   ``--log-dir`` (matching ``runner.logger.log_path`` in the PPO config) so
   they show up alongside RL training curves.
3. Optionally writes the markdown report next to the candidate dataset
   for human review.

Usage::

    python -m src.rl.eval_with_diagnostics \\
        --ref-dataset /home/hlei/.cache/huggingface/lerobot/EverNorif/leisaac-pick-orange \\
        --cand-dataset ./logs/pick_orange_ppo_eval/rollout/global_step_100/dataset \\
        --log-dir ./logs/pick_orange_ppo \\
        --global-step 100

Watch mode polls the parent ``--watch-dir`` for new
``global_step_*/dataset`` subdirs and processes each exactly once::

    python -m src.rl.eval_with_diagnostics --watch \\
        --watch-dir ./logs/pick_orange_ppo/eval_rollouts \\
        --ref-dataset .../leisaac-pick-orange --log-dir ./logs/pick_orange_ppo
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterable

# Ensure src.diagnostics is importable when run as a script.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.diagnostics.online_callback import (  # noqa: E402
    DEFAULT_MODULES,
    run_diagnostics_on_rollout_dir,
)
from src.diagnostics.report import render  # noqa: E402

_STEP_RE = re.compile(r"global_step_(\d+)")


def _write_tensorboard(
    log_dir: Path,
    flat: dict[str, float],
    global_step: int,
    tag_prefix: str = "eval/diag",
) -> None:
    """Best-effort TB writer. Soft-fails if torch.utils.tensorboard isn't installed."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as e:
        print(f"[warn] tensorboard unavailable, skipping write: {e}", file=sys.stderr)
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    try:
        for k, v in flat.items():
            writer.add_scalar(f"{tag_prefix}/{k}", v, global_step=global_step)
    finally:
        writer.flush()
        writer.close()


def _run_once(
    ref_dataset: Path,
    cand_dataset: Path,
    log_dir: Path,
    global_step: int,
    modules: Iterable[str] = DEFAULT_MODULES,
    write_markdown: bool = True,
) -> dict[str, float]:
    flat, results = run_diagnostics_on_rollout_dir(
        ref_root=ref_dataset, cand_root=cand_dataset, modules=modules,
    )
    _write_tensorboard(log_dir, flat, global_step=global_step)

    out_dir = cand_dataset.parent
    (out_dir / "diagnostics.json").write_text(
        json.dumps(
            {"global_step": global_step, "metrics": flat,
             "results": [r.to_dict() for r in results]},
            indent=2, ensure_ascii=False,
        )
    )
    if write_markdown:
        try:
            (out_dir / "diagnostics.md").write_text(render(results))
        except RuntimeError as e:
            print(f"[warn] markdown render failed: {e}", file=sys.stderr)

    print(f"[step {global_step}] wrote {len(flat)} scalars to {log_dir}")
    for k, v in sorted(flat.items()):
        print(f"  {k} = {v:.6f}")
    return flat


def _parse_step_from_path(p: Path) -> int | None:
    for part in (p.name, *(parent.name for parent in p.parents)):
        m = _STEP_RE.search(part)
        if m:
            return int(m.group(1))
    return None


def _iter_new_rollouts(watch_dir: Path, seen: set[Path]) -> list[tuple[int, Path]]:
    """Find ``watch_dir/global_step_*/dataset`` directories not yet processed."""
    out: list[tuple[int, Path]] = []
    if not watch_dir.is_dir():
        return out
    for step_dir in sorted(watch_dir.glob("global_step_*")):
        cand = step_dir / "dataset"
        if cand in seen or not (cand / "meta" / "info.json").exists():
            continue
        step = _parse_step_from_path(step_dir) or 0
        out.append((step, cand))
        seen.add(cand)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser("python -m src.rl.eval_with_diagnostics")
    p.add_argument("--ref-dataset", type=Path, required=True,
                   help="LeRobot v2.1 reference dataset root (demo set).")
    p.add_argument("--cand-dataset", type=Path,
                   help="LeRobot v2.1 candidate dataset root. Required unless --watch.")
    p.add_argument("--log-dir", type=Path, required=True,
                   help="TensorBoard log dir; should equal runner.logger.log_path "
                        "in the PPO config so scalars land in the same event file.")
    p.add_argument("--global-step", type=int, default=0,
                   help="Step tag for the TB scalars. Auto-parsed from "
                        "'global_step_N' in --cand-dataset path if 0.")
    p.add_argument("--modules", type=str, default="",
                   help="Comma-separated diagnostic names; empty → DEFAULT_MODULES.")
    p.add_argument("--watch", action="store_true",
                   help="Poll --watch-dir for new global_step_*/dataset/ subdirs.")
    p.add_argument("--watch-dir", type=Path,
                   help="Parent dir to poll in --watch mode.")
    p.add_argument("--watch-interval-s", type=float, default=30.0)
    p.add_argument("--no-markdown", action="store_true")
    args = p.parse_args(argv)

    modules = (
        tuple(s.strip() for s in args.modules.split(",") if s.strip())
        or DEFAULT_MODULES
    )

    if args.watch:
        if args.watch_dir is None:
            p.error("--watch requires --watch-dir")
        seen: set[Path] = set()
        print(f"watching {args.watch_dir} every {args.watch_interval_s}s; Ctrl-C to stop")
        while True:
            for step, cand in _iter_new_rollouts(args.watch_dir, seen):
                try:
                    _run_once(
                        ref_dataset=args.ref_dataset,
                        cand_dataset=cand,
                        log_dir=args.log_dir,
                        global_step=step,
                        modules=modules,
                        write_markdown=not args.no_markdown,
                    )
                except Exception as exc:  # don't crash the watcher on one bad rollout
                    print(f"[error] step {step} dataset {cand}: {exc}", file=sys.stderr)
            time.sleep(args.watch_interval_s)

    if args.cand_dataset is None:
        p.error("--cand-dataset is required unless --watch")
    step = args.global_step or _parse_step_from_path(args.cand_dataset) or 0
    _run_once(
        ref_dataset=args.ref_dataset,
        cand_dataset=args.cand_dataset,
        log_dir=args.log_dir,
        global_step=step,
        modules=modules,
        write_markdown=not args.no_markdown,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
