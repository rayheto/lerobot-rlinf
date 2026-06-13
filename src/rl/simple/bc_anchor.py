"""Behavioral-cloning anchor over the EverNorif/leisaac-pick-orange demo set.

Goal: prevent the residual head from drifting away from SFT behavior on demo-
distribution states. Uses an SFT-converged approximation:

  On demo (state, a_demo), pi05's base prediction ≈ a_demo (because pi05 was
  SFT'd on exactly these pairs). Therefore the *desired residual* on demo
  states is ≈ 0. We feed `base := a_demo` as a self-consistent stand-in for
  the head's "base" input, so no pi05 forward is needed during the BC step.

  BC loss := -E[ logp(r=0 | head(state, a_demo)) ]

This encourages μ_residual → 0 and σ → small on demo states, both via the
Gaussian NLL. Combined with PPO advantage gradient that pushes the residual
to be non-zero where the env rewards correction, BC acts as a soft anchor.

Trade-off: BC sees the head with `base=a_demo`, but at rollout the head sees
`base=pi05(obs)`. These distributions match iff SFT is tight (EXP_01 = 0.726
is on the WARNING boundary, so the approximation degrades modestly there).
"""
from __future__ import annotations

import pathlib

import numpy as np
import torch

from src.diagnostics.io import iter_episode_parquets, load_episode_parquet


def _load_demo_pairs(root: str) -> tuple[np.ndarray, np.ndarray]:
    states_parts: list[np.ndarray] = []
    actions_parts: list[np.ndarray] = []
    for p in iter_episode_parquets(pathlib.Path(root).expanduser()):
        cols = load_episode_parquet(p, columns=["observation.state", "action"])
        s = cols["observation.state"]
        a = cols["action"]
        if s.ndim != 2 or a.ndim != 2 or s.shape[0] != a.shape[0]:
            continue
        states_parts.append(s.astype(np.float32))
        actions_parts.append(a.astype(np.float32))
    if not states_parts:
        raise RuntimeError(f"no usable (state, action) pairs under {root}")
    return np.concatenate(states_parts, axis=0), np.concatenate(actions_parts, axis=0)


class BCAnchor:
    """In-memory (state, action) cache + sampler + BC loss."""

    def __init__(self, demo_dataset_path: str, batch_size: int, device: str = "cuda"):
        states, actions = _load_demo_pairs(demo_dataset_path)
        self.states = torch.as_tensor(states, dtype=torch.float32, device=device)
        self.actions = torch.as_tensor(actions, dtype=torch.float32, device=device)
        self.batch_size = int(batch_size)
        self.N = int(self.states.shape[0])
        self.device = device

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, self.N, (self.batch_size,), device=self.device)
        return self.states[idx], self.actions[idx]

    def loss(self, policy) -> torch.Tensor:
        """BC loss = -mean( logp(residual=0 | head(state, a_demo)) ).

        Uses ``a_demo`` as the head's `base` input (SFT-converged stand-in).
        ``actions == base_actions`` makes ``r = action - base = 0`` inside
        the policy's evaluate(), so we get logp(0 residual) directly.
        """
        states, actions = self.sample()
        logp, _, _ = policy.evaluate(states, actions, actions)
        return -logp.mean()
