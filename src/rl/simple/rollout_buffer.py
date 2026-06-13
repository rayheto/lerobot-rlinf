"""Fixed-length rollout buffer with GAE.

Stored per-step (T, Nenv, ...):
  state         (Nenv, state_dim)     — used for head input + OOD penalty
  base_action   (Nenv, act_dim)       — deterministic pi05 chunk[0]
  action        (Nenv, act_dim)       — base + sampled residual (the executed action)
  logp_old     (Nenv,)                — log π(action | state) at rollout time
  value_old    (Nenv,)                — V(state) at rollout time
  reward       (Nenv,)                — shaped reward
  done         (Nenv,)                — terminal OR truncation (for GAE bootstrapping)

Images are NOT stored — we only re-evaluate the residual head and value head
during PPO updates, neither of which needs pixels. The bootstrap V(s_T) is
computed externally and passed into compute_gae().
"""
from __future__ import annotations

from typing import Iterator

import torch


class RolloutBuffer:
    def __init__(
        self,
        T: int,
        Nenv: int,
        state_dim: int,
        act_dim: int,
        device: str = "cuda",
    ):
        self.T = T
        self.Nenv = Nenv
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.device = device

        dev = device
        f32 = torch.float32
        self.states = torch.zeros(T, Nenv, state_dim, device=dev, dtype=f32)
        self.base_actions = torch.zeros(T, Nenv, act_dim, device=dev, dtype=f32)
        self.actions = torch.zeros(T, Nenv, act_dim, device=dev, dtype=f32)
        self.logp_old = torch.zeros(T, Nenv, device=dev, dtype=f32)
        self.value_old = torch.zeros(T, Nenv, device=dev, dtype=f32)
        self.rewards = torch.zeros(T, Nenv, device=dev, dtype=f32)
        self.dones = torch.zeros(T, Nenv, device=dev, dtype=f32)  # float for math
        self.advantages = torch.zeros(T, Nenv, device=dev, dtype=f32)
        self.returns = torch.zeros(T, Nenv, device=dev, dtype=f32)

        self._t = 0

    def reset(self) -> None:
        self._t = 0

    @property
    def full(self) -> bool:
        return self._t >= self.T

    def add(
        self,
        state: torch.Tensor,
        base_action: torch.Tensor,
        action: torch.Tensor,
        logp: torch.Tensor,
        value: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        if self._t >= self.T:
            raise RuntimeError("RolloutBuffer.add called after buffer is full")
        t = self._t
        self.states[t].copy_(state.detach().to(self.device, torch.float32))
        self.base_actions[t].copy_(base_action.detach().to(self.device, torch.float32))
        self.actions[t].copy_(action.detach().to(self.device, torch.float32))
        self.logp_old[t].copy_(logp.detach().to(self.device, torch.float32))
        self.value_old[t].copy_(value.detach().to(self.device, torch.float32))
        self.rewards[t].copy_(reward.detach().to(self.device, torch.float32))
        self.dones[t].copy_(done.detach().to(self.device, torch.float32))
        self._t += 1

    def compute_gae(
        self,
        gamma: float,
        lam: float,
        last_value: torch.Tensor,
        normalize: bool = True,
    ) -> None:
        """Standard GAE-λ. last_value: (Nenv,) bootstrap V(s_T)."""
        T = self.T
        last_value = last_value.detach().to(self.device, torch.float32)
        gae = torch.zeros(self.Nenv, device=self.device, dtype=torch.float32)

        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else self.value_old[t + 1]
            not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * not_done - self.value_old[t]
            gae = delta + gamma * lam * not_done * gae
            self.advantages[t].copy_(gae)

        self.returns.copy_(self.advantages + self.value_old)

        if normalize:
            adv = self.advantages
            mean = adv.mean()
            std = adv.std().clamp_min(1e-6)
            self.advantages.copy_((adv - mean) / std)

    def iter_minibatches(
        self,
        minibatch_size: int,
        num_epochs: int,
    ) -> Iterator[dict]:
        """Yield flattened (T*Nenv, ...) minibatches over `num_epochs` passes."""
        N = self.T * self.Nenv
        flat_states = self.states.reshape(N, self.state_dim)
        flat_base = self.base_actions.reshape(N, self.act_dim)
        flat_act = self.actions.reshape(N, self.act_dim)
        flat_logp = self.logp_old.reshape(N)
        flat_val = self.value_old.reshape(N)
        flat_adv = self.advantages.reshape(N)
        flat_ret = self.returns.reshape(N)

        for _ in range(num_epochs):
            perm = torch.randperm(N, device=self.device)
            for start in range(0, N, minibatch_size):
                idx = perm[start : start + minibatch_size]
                yield {
                    "states": flat_states[idx],
                    "base_actions": flat_base[idx],
                    "actions": flat_act[idx],
                    "logp_old": flat_logp[idx],
                    "value_old": flat_val[idx],
                    "advantages": flat_adv[idx],
                    "returns": flat_ret[idx],
                }
