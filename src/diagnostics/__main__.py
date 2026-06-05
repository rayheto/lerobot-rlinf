from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import REGISTRY, run_all
from .base import DiagnosticContext
from .io import load_info
from .registry import names as registry_names
from .report import render_to_files
from .result import Status


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("python -m src.diagnostics")
    p.add_argument("--ref", type=Path, help="LeRobot reference dataset root")
    p.add_argument("--cand", type=Path, help="LeRobot candidate dataset root")
    p.add_argument(
        "--modules", type=str, default="",
        help="comma-separated diagnostic names; empty → run all registered",
    )
    p.add_argument("--meta-json", type=Path, help="JSON file with abstract metadata")
    p.add_argument("--out-json", type=Path, help="path to write results JSON")
    p.add_argument("--out-md", type=Path, help="path to write Markdown report")
    p.add_argument("--list", action="store_true", help="list registered diagnostics and exit")
    p.add_argument("--selftest", action="store_true", help="run end-to-end selftest")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.list:
        for n in registry_names():
            cls = REGISTRY[n]
            print(f"{n}\t{cls.category}\t{cls.thresholds}")
        return 0

    if args.selftest:
        return _selftest()

    if not args.ref or not args.cand:
        print("--ref and --cand are required (or use --selftest / --list)", file=sys.stderr)
        return 2

    meta = json.loads(args.meta_json.read_text()) if args.meta_json else None
    ctx = DiagnosticContext(
        ref_root=args.ref,
        cand_root=args.cand,
        ref_info=load_info(args.ref),
        cand_info=load_info(args.cand),
    )
    names = [s.strip() for s in args.modules.split(",") if s.strip()] or None
    results = run_all(ctx, names=names)
    render_to_files(results, args.out_json, args.out_md, meta=meta)

    worst = max(
        (r.status for r in results),
        key=lambda s: ["OK", "SKIPPED", "WARNING", "CRITICAL", "ERROR"].index(s.value),
        default=Status.OK,
    )
    for r in results:
        line = f"[{r.status.value:8s}] {r.name}"
        if r.metrics:
            line += "  " + " ".join(f"{k}={v}" for k, v in r.metrics.items())
        print(line)
    return 0 if worst in (Status.OK, Status.WARNING) else 1


# --------------------------------------------------------------------------- #
# Synthetic LeRobot v2.x dataset writer (selftest only)
# --------------------------------------------------------------------------- #

_FEATURE_DIM = 6
_FPS = 30


def _info_payload(total_episodes: int, total_frames: int) -> dict:
    return {
        "codebase_version": "v2.1",
        "robot_type": "synthetic_diag",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": _FPS,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "action": {
                "dtype": "float32", "shape": [_FEATURE_DIM],
                "names": [f"j{i}.pos" for i in range(_FEATURE_DIM)],
            },
            "observation.state": {
                "dtype": "float32", "shape": [_FEATURE_DIM],
                "names": [f"j{i}.pos" for i in range(_FEATURE_DIM)],
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _episode_stats(ep_idx: int, action: np.ndarray, state: np.ndarray) -> dict:
    T = action.shape[0]
    return {
        "episode_index": ep_idx,
        "stats": {
            "action": {
                "min": action.min(axis=0).tolist(),
                "max": action.max(axis=0).tolist(),
                "mean": action.mean(axis=0).tolist(),
                "std": action.std(axis=0).tolist(),
                "count": [int(T)],
            },
            "observation.state": {
                "min": state.min(axis=0).tolist(),
                "max": state.max(axis=0).tolist(),
                "mean": state.mean(axis=0).tolist(),
                "std": state.std(axis=0).tolist(),
                "count": [int(T)],
            },
        },
    }


def _write_episode_parquet(
    root: Path, ep_idx: int, global_offset: int,
    action: np.ndarray, state: np.ndarray,
) -> None:
    T = action.shape[0]
    chunk_dir = root / "data" / "chunk-000"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "action": pa.array(action.astype(np.float32).tolist(),
                               type=pa.list_(pa.float32())),
            "observation.state": pa.array(state.astype(np.float32).tolist(),
                                          type=pa.list_(pa.float32())),
            "timestamp": pa.array(
                np.arange(T, dtype=np.float32) / _FPS, type=pa.float32()
            ),
            "frame_index": pa.array(np.arange(T, dtype=np.int64), type=pa.int64()),
            "episode_index": pa.array(
                np.full(T, ep_idx, dtype=np.int64), type=pa.int64()
            ),
            "index": pa.array(
                np.arange(global_offset, global_offset + T, dtype=np.int64),
                type=pa.int64(),
            ),
            "task_index": pa.array(np.zeros(T, dtype=np.int64), type=pa.int64()),
        }
    )
    pq.write_table(table, chunk_dir / f"episode_{ep_idx:06d}.parquet")


def _synthesize_dataset(
    root: Path, n_episodes: int, n_frames: int,
    action_std: float, state_step_std: float, seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    episodes_rows: list[dict] = []
    stats_rows: list[dict] = []
    global_offset = 0
    for ep in range(n_episodes):
        action = rng.normal(0.0, action_std, size=(n_frames, _FEATURE_DIM))
        deltas = rng.normal(0.0, state_step_std, size=(n_frames - 1, _FEATURE_DIM))
        state = np.concatenate([np.zeros((1, _FEATURE_DIM)), np.cumsum(deltas, axis=0)], axis=0)
        _write_episode_parquet(root, ep, global_offset, action, state)
        episodes_rows.append({"episode_index": ep, "tasks": ["selftest"], "length": int(n_frames)})
        stats_rows.append(_episode_stats(ep, action, state))
        global_offset += n_frames

    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(_info_payload(n_episodes, n_episodes * n_frames), indent=2)
    )
    _write_jsonl(root / "meta" / "episodes.jsonl", episodes_rows)
    _write_jsonl(root / "meta" / "episodes_stats.jsonl", stats_rows)
    _write_jsonl(root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": "selftest"}])


def _selftest() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="diag_selftest_"))
    ref_root = tmp / "ref"
    cand_root = tmp / "cand"
    n_eps, n_frames = 5, 200

    action_std_ref = 1.0
    action_std_cand = 0.4 * action_std_ref     # → EXP_01 ratio ≈ 0.4 (CRITICAL)
    state_step_ref = 0.2
    state_step_cand = 1.85 * state_step_ref    # → EXP_02 path_len_ratio ≈ 1.85 (WARNING)

    _synthesize_dataset(ref_root, n_eps, n_frames,
                        action_std=action_std_ref, state_step_std=state_step_ref, seed=0)
    _synthesize_dataset(cand_root, n_eps, n_frames,
                        action_std=action_std_cand, state_step_std=state_step_cand, seed=1)

    ctx = DiagnosticContext(
        ref_root=ref_root, cand_root=cand_root,
        ref_info=load_info(ref_root), cand_info=load_info(cand_root),
    )
    results = run_all(ctx)
    by = {r.name: r for r in results}

    failures: list[str] = []

    r1 = by["EXP_01_Mode_Averaging"]
    if r1.status != Status.CRITICAL:
        failures.append(f"EXP_01 status expected CRITICAL, got {r1.status.value}")
    if not (0.30 <= r1.metrics["ratio"] <= 0.50):
        failures.append(f"EXP_01 ratio out of band: {r1.metrics['ratio']}")

    r2 = by["EXP_02_Compounding_Error"]
    if r2.status != Status.WARNING:
        failures.append(f"EXP_02 status expected WARNING, got {r2.status.value}")
    if not (1.65 <= r2.metrics["path_len_ratio"] <= 2.05):
        failures.append(f"EXP_02 path_len_ratio out of band: {r2.metrics['path_len_ratio']}")

    meta = {
        "task": "selftest",
        "num_demos": n_eps,
        "success_rate_range": "n/a",
        "temporal_inflation_range": "n/a",
    }
    out_json = tmp / "results.json"
    out_md = tmp / "report.md"
    render_to_files(results, out_json, out_md, meta=meta)
    md = out_md.read_text()
    for required in ("## Abstract", "## EXP_01", "## EXP_02", "## Discussion"):
        if required not in md:
            failures.append(f"report missing section: {required}")

    print(f"selftest artifacts: {tmp}")
    for r in results:
        print(f"  [{r.status.value:8s}] {r.name}  {r.metrics}")

    if failures:
        for line in failures:
            print(f"FAIL  {line}", file=sys.stderr)
        return 1
    print("selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
