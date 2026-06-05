from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow.parquet as pq


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_info(root: Path) -> dict[str, Any]:
    with (root / "meta" / "info.json").open() as f:
        return json.load(f)


def load_episodes(root: Path) -> list[dict[str, Any]]:
    return _read_jsonl(root / "meta" / "episodes.jsonl")


def load_episodes_stats(root: Path) -> list[dict[str, Any]] | None:
    path = root / "meta" / "episodes_stats.jsonl"
    if not path.exists():
        return None
    return _read_jsonl(path)


def iter_episode_parquets(root: Path) -> Iterator[Path]:
    data_dir = root / "data"
    yield from sorted(data_dir.glob("chunk-*/episode_*.parquet"))


def load_episode_parquet(path: Path, columns: list[str] | None = None) -> dict[str, np.ndarray]:
    table = pq.read_table(path, columns=columns)
    out: dict[str, np.ndarray] = {}
    for col in table.column_names:
        arr = table.column(col).to_pylist()
        # parquet list<float> columns come back as list-of-lists -> stack
        if arr and isinstance(arr[0], list):
            out[col] = np.asarray(arr, dtype=np.float32)
        else:
            out[col] = np.asarray(arr)
    return out
