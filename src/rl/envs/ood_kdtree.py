"""KNN OOD penalty backing r_ood in the PickOrange RL reward.

Loads the SO-101 pick-orange demo states from a LeRobot v2.x dataset, builds
a KD-tree, and exposes a per-step query that returns d_norm = k-NN mean L2
distance divided by sigma. Sigma uses the same intra-reference half-vs-half
1-NN-mean as EXP_05 (src/diagnostics/modules/state_coverage.py), so the
reward term and the diagnostic are numerically aligned: r_ood ≈ -coverage_ratio
on a held-out trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Iterable

import numpy as np
import torch
from scipy.spatial import cKDTree

from src.diagnostics.io import iter_episode_parquets, load_episode_parquet

_FEATURE = "observation.state"
_SEED = 0


def _collect_demo_states(root: Path) -> np.ndarray:
    parts: list[np.ndarray] = []
    for p in iter_episode_parquets(root):
        cols = load_episode_parquet(p, columns=[_FEATURE])
        x = cols[_FEATURE]
        if x.ndim == 2 and x.shape[0] >= 1:
            parts.append(x.astype(np.float64))
    if not parts:
        raise RuntimeError(f"no usable {_FEATURE} under {root}")
    return np.concatenate(parts, axis=0)


def _intra_ref_sigma(points: np.ndarray, seed: int = _SEED + 2) -> float:
    # Half-vs-half 1-NN mean — same convention as EXP_05's intra_ref_nn_mean.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(points.shape[0])
    half = points.shape[0] // 2
    a, b = points[perm[:half]], points[perm[half : 2 * half]]
    tree = cKDTree(b)
    d, _ = tree.query(a, k=1)
    return float(np.mean(d))


@dataclass
class OodKNNConfig:
    demo_dataset_path: str
    k_neighbors: int = 5
    coef: float = 1.0


class OodKNNPenalty:
    """KD-tree backed OOD penalty. One process-wide instance per dataset path."""

    _instances: dict[str, "OodKNNPenalty"] = {}
    _lock = Lock()

    def __init__(self, cfg: OodKNNConfig):
        root = Path(cfg.demo_dataset_path).expanduser()
        states = _collect_demo_states(root)
        self.k = int(cfg.k_neighbors)
        self.coef = float(cfg.coef)
        self.sigma = _intra_ref_sigma(states)
        if self.sigma <= 0.0:
            raise RuntimeError(
                f"intra-ref sigma is non-positive ({self.sigma}); demo set degenerate"
            )
        self.tree = cKDTree(states)
        self.n_points = int(states.shape[0])
        self.dim = int(states.shape[1])

    @classmethod
    def get(cls, cfg: OodKNNConfig) -> "OodKNNPenalty":
        key = str(Path(cfg.demo_dataset_path).expanduser().resolve())
        with cls._lock:
            inst = cls._instances.get(key)
            if inst is None or inst.k != cfg.k_neighbors:
                inst = cls(cfg)
                cls._instances[key] = inst
            inst.coef = float(cfg.coef)
            return inst

    def query_d_norm(self, joint_pos: Iterable[float] | np.ndarray | torch.Tensor) -> np.ndarray:
        """Return d_norm = mean k-NN distance / sigma, shape (N,)."""
        if isinstance(joint_pos, torch.Tensor):
            q = joint_pos.detach().cpu().numpy().astype(np.float64, copy=False)
        else:
            q = np.asarray(joint_pos, dtype=np.float64)
        if q.ndim == 1:
            q = q[None, :]
        d, _ = self.tree.query(q, k=self.k)
        if self.k == 1:
            d = d[:, None]
        return d.mean(axis=1) / self.sigma

    def reward(self, joint_pos: torch.Tensor) -> torch.Tensor:
        """Return r_ood = -coef * d_norm as a torch tensor on joint_pos.device."""
        d_norm = self.query_d_norm(joint_pos)
        return torch.as_tensor(-self.coef * d_norm, dtype=torch.float32, device=joint_pos.device)
