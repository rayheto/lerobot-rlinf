"""ResidualGaussianPolicy: pi05 (JAX, frozen) + small Gaussian residual head + value head.

Action pipeline at rollout time (chunk_execute mode — the *only* mode where
the SFT 24999 ckpt visibly grasps, as validated against src/eval.py):

  base[t]   = chunk[step_in_chunk]                  # consumed step-by-step
  μ, lσ     = head([state, base])
  r ~ N(μ, σ²)                                      # PPO exploration noise
  action    = base + r                              # executed
  logp      = sum(Normal(μ, σ).log_prob(r))

Each env keeps its own chunk cache + step_in_chunk counter. We re-forward
pi05 only when (a) the cache is empty (initial step / reset), or (b)
``step_in_chunk >= len(chunk)`` (10-step chunk exhausted), or (c) the caller
passes ``reset_mask[i] = True`` (env i was reset between calls).

Why this matches SFT behavior: pi05 was trained on coherent 10-step chunks.
Re-forwarding every step and only using chunk[0] (the previous design)
effectively pins the policy to the "first frame of a chunk" distribution
forever, which is degenerate. With chunk_execute the video shows normal
motor behavior; without it the same ckpt + env produces random jitter — only
diff is chunk handling.

PPO evaluate-time: head is replayed on STORED (state_t, base_t). pi05 is not
re-queried, because base_t is whatever chunk slot was consumed at step t —
already in the buffer.

Why no image into the head: pi05's `base` already encodes its visual reading.
The head's job is "given the SFT chunk slot and current joint state, output a
small corrective residual" — primarily to combat OOD-stall (EXP_05).

Why deterministic pi05: PPO needs ``action - base = residual`` to be the exact
sample drawn from the policy distribution. Stochastic pi05 noise would break
that identity at update time. (For the JAX server, deterministic-ish behavior
comes from the server's default sampling config — pi05 uses very low noise.)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Base interface — keeps PPOTrainer/Buffer agnostic to policy implementation.
# ---------------------------------------------------------------------------


class BasePolicy(nn.Module, ABC):
    """Abstract policy interface for the simple PPO trainer."""

    @abstractmethod
    def act(
        self,
        obs: Dict[str, torch.Tensor],
        reset_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (action, base_action, logp, value).  All (Nenv, ...)."""

    @abstractmethod
    def evaluate(
        self,
        states: torch.Tensor,
        base_actions: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate stored transitions. Returns (logp, value, entropy)."""

    @abstractmethod
    def value(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Bootstrap V(s_T) for GAE. Shape (Nenv,)."""

    @abstractmethod
    def trainable_params(self):
        ...

    def policy_params(self):
        """Override to return only policy (action-head) params for split grad clip."""
        return self.trainable_params()

    def value_params(self):
        """Override to return only value-head params for split grad clip."""
        return []


# ---------------------------------------------------------------------------
# pi05 JAX server backend with per-env chunk cache.
# ---------------------------------------------------------------------------


class _Pi05JaxWrapper:
    """Per-env chunk-execute wrapper around an openpi JAX server.

    The server itself is spawned/killed by the caller (see _openpi_server.py
    helpers used in train.py). This class only owns the client side: socket,
    per-env chunk caches, step counters.
    """

    def __init__(
        self,
        host: str,
        port: int,
        prompt: str,
        chunk_horizon: int,
        device: str = "cuda",
    ):
        from src.rl.simple._openpi_server import ensure_openpi_client_on_path

        ensure_openpi_client_on_path()
        from openpi_client import image_tools  # noqa: E402
        from openpi_client.websocket_client_policy import (  # noqa: E402
            WebsocketClientPolicy,
        )

        self._image_tools = image_tools
        self._client = WebsocketClientPolicy(host=host, port=port)
        self._metadata = self._client.get_server_metadata()
        print(
            f"[pi05-jax] connected ws://{host}:{port}  meta={self._metadata}",
            flush=True,
        )

        self.device = device
        self.prompt = prompt
        self.action_horizon = int(chunk_horizon)

        # Per-env state — sized lazily on first call when we learn Nenv.
        self._chunk_cache: List[Optional[np.ndarray]] = []
        self._step_in_chunk: List[int] = []

    # ------------------------- per-env cache plumbing -------------------------

    def _ensure_cache(self, nenv: int) -> None:
        if len(self._chunk_cache) != nenv:
            self._chunk_cache = [None] * nenv
            self._step_in_chunk = [0] * nenv

    @staticmethod
    def _to_hwc_uint8(x) -> np.ndarray:
        """Normalize env image obs (tensor or array, CHW or HWC, float or uint8) to HWC uint8."""
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
            x = np.transpose(x, (1, 2, 0))
        if x.dtype != np.uint8:
            xmax = float(x.max()) if x.size else 1.0
            if xmax <= 1.5:
                x = (x * 255.0).clip(0, 255).astype(np.uint8)
            else:
                x = x.clip(0, 255).astype(np.uint8)
        if x.ndim == 2:
            x = np.stack([x, x, x], axis=-1)
        if x.shape[-1] == 1:
            x = np.repeat(x, 3, axis=-1)
        return x

    def _build_client_obs_i(self, obs: Dict[str, torch.Tensor], i: int) -> dict:
        front = self._to_hwc_uint8(obs["main_images"][i])
        wrist = self._to_hwc_uint8(obs["wrist_images"][i])
        state_motor_deg = obs["states"][i].detach().cpu().numpy().astype(np.float64)
        prompts = obs.get("task_descriptions")
        if prompts is None:
            prompt = self.prompt
        else:
            prompt = prompts[i] if isinstance(prompts, (list, tuple)) else str(prompts[i])
        return {
            "images/front": self._image_tools.convert_to_uint8(
                self._image_tools.resize_with_pad(front, 224, 224)
            ),
            "images/wrist": self._image_tools.convert_to_uint8(
                self._image_tools.resize_with_pad(wrist, 224, 224)
            ),
            "state": state_motor_deg,
            "prompt": prompt,
        }

    def _refill_i(self, obs: Dict[str, torch.Tensor], i: int) -> None:
        client_obs = self._build_client_obs_i(obs, i)
        chunk = self._client.infer(client_obs)["actions"]  # (T, 6) motor-deg
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[-1] != 6:
            raise RuntimeError(f"unexpected chunk shape from server: {chunk.shape}")
        self._chunk_cache[i] = chunk
        self._step_in_chunk[i] = 0

    # ------------------------- public API -------------------------

    @torch.no_grad()
    def base_action(
        self,
        obs: Dict[str, torch.Tensor],
        reset_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Consume next chunk slot for each env. Refills via the server when
        the cache is empty, exhausted, or the env was just reset.

        reset_mask: (Nenv,) bool/int tensor. True ⇒ force re-infer for env i.
        """
        nenv = obs["states"].shape[0]
        self._ensure_cache(nenv)

        if reset_mask is None:
            rm = [False] * nenv
        else:
            rm = [bool(x) for x in reset_mask.detach().cpu().tolist()]

        bases = np.empty((nenv, 6), dtype=np.float32)
        for i in range(nenv):
            cache = self._chunk_cache[i]
            need = (
                cache is None
                or rm[i]
                or self._step_in_chunk[i] >= cache.shape[0]
            )
            if need:
                self._refill_i(obs, i)
                cache = self._chunk_cache[i]
            step = self._step_in_chunk[i]
            bases[i] = cache[step]
            self._step_in_chunk[i] = step + 1
        return torch.as_tensor(bases, device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def peek_base_action(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the *next* base slot per env WITHOUT advancing the counter.

        Used by ``value(obs)`` for GAE bootstrap V(s_T): we need a base to feed
        the value head but mustn't consume a chunk slot that a future act()
        will replay. If the cache is empty/exhausted we refill (so next act
        starts from step=0 on the same chunk we just minted).
        """
        nenv = obs["states"].shape[0]
        self._ensure_cache(nenv)
        bases = np.empty((nenv, 6), dtype=np.float32)
        for i in range(nenv):
            cache = self._chunk_cache[i]
            if cache is None or self._step_in_chunk[i] >= cache.shape[0]:
                self._refill_i(obs, i)
                cache = self._chunk_cache[i]
            step = self._step_in_chunk[i]
            bases[i] = cache[step]
        return torch.as_tensor(bases, device=self.device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Residual heads
# ---------------------------------------------------------------------------


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


class _ResidualHead(nn.Module):
    """Outputs μ_residual (act_dim) and log_σ (act_dim) from [state, base]."""

    def __init__(self, in_dim: int, hidden: int, act_dim: int, init_log_std: float):
        super().__init__()
        self.act_dim = act_dim
        self.trunk = _mlp(in_dim, hidden, 2 * act_dim)
        # Zero-out the final layer so μ_residual ≈ 0 and log_σ ≈ init_log_std
        # at start of training (head behaves as identity on top of pi05).
        last = self.trunk[-1]
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias.zero_()
            last.bias[act_dim:].fill_(init_log_std)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.trunk(h)
        mu, log_sigma = out.chunk(2, dim=-1)
        return mu, log_sigma


class _ValueHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.trunk = _mlp(in_dim, hidden, 1)
        # Zero-init the last layer so V(s)≈0 at start of training. Without this,
        # random V on returns ~O(±50) yields value_loss ~O(1000) → grad_norm
        # ~O(1000s) dominates the shared grad clip pool, contaminating the
        # policy update. See docs/dryrun_jax_step8f_crossanalysis.md §6.
        last = self.trunk[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.trunk(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Main policy
# ---------------------------------------------------------------------------


class ResidualGaussianPolicy(BasePolicy):
    """pi05 frozen (JAX server) + Gaussian residual head + value head."""

    STATE_DIM = 6
    ACT_DIM = 6

    def __init__(self, cfg, device: str = "cuda"):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.pi05 = _Pi05JaxWrapper(
            host=cfg.pi05_server_host,
            port=cfg.pi05_server_port,
            prompt=cfg.env_prompt,
            chunk_horizon=cfg.pi05_chunk_horizon,
            device=device,
        )

        head_in = self.STATE_DIM + self.ACT_DIM  # = 12
        self.head = _ResidualHead(
            in_dim=head_in,
            hidden=cfg.head_hidden,
            act_dim=self.ACT_DIM,
            init_log_std=cfg.head_init_log_std,
        ).to(device)
        self.value_head = _ValueHead(
            in_dim=head_in,
            hidden=cfg.head_hidden,
        ).to(device)

    # ------------------------- helpers -------------------------

    def _head_input(self, states: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [states.to(self.device, torch.float32), base.to(self.device, torch.float32)],
            dim=-1,
        )

    def _dist_from_head(self, h: torch.Tensor) -> torch.distributions.Independent:
        mu, log_sigma = self.head(h)
        log_sigma = log_sigma.clamp(self.cfg.log_std_min, self.cfg.log_std_max)
        sigma = log_sigma.exp()
        base = torch.distributions.Normal(mu, sigma)
        return torch.distributions.Independent(base, 1)

    # ------------------------- BasePolicy API -------------------------

    @torch.no_grad()
    def act(
        self,
        obs: Dict[str, torch.Tensor],
        reset_mask: Optional[torch.Tensor] = None,
    ):
        base = self.pi05.base_action(obs, reset_mask=reset_mask)  # (Nenv, 6)
        states = obs["states"].to(self.device, torch.float32)
        h = self._head_input(states, base)
        dist = self._dist_from_head(h)
        r = dist.sample()
        if self.cfg.residual_clip > 0:
            r = r.clamp(-self.cfg.residual_clip, self.cfg.residual_clip)
        action = base + r
        logp = dist.log_prob(r)
        value = self.value_head(h)
        return action, base, logp, value

    def evaluate(self, states, base_actions, actions):
        h = self._head_input(states, base_actions)
        dist = self._dist_from_head(h)
        r = actions - base_actions
        logp = dist.log_prob(r)
        entropy = dist.entropy()
        value = self.value_head(h)
        return logp, value, entropy

    @torch.no_grad()
    def value(self, obs):
        base = self.pi05.peek_base_action(obs)  # don't consume a chunk slot
        states = obs["states"].to(self.device, torch.float32)
        h = self._head_input(states, base)
        return self.value_head(h)

    def trainable_params(self):
        return list(self.head.parameters()) + list(self.value_head.parameters())

    def policy_params(self):
        return list(self.head.parameters())

    def value_params(self):
        return list(self.value_head.parameters())
