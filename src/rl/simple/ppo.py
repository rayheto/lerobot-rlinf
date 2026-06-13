"""PPO trainer for the residual Gaussian policy.

Loss components per minibatch:
  L_pg     = clipped surrogate (single-direction PPO)
  L_value  = MSE on clipped value prediction
  L_ent    = -entropy bonus
  L_bc     = BC anchor (on demo (state, action) pairs)
  L        = L_pg + value_coef · L_value - ent_coef · L_ent_mean + bc_coef · L_bc

Approx KL is computed each minibatch from logp_new − logp_old; if it exceeds
``cfg.target_kl`` we early-stop the inner epoch loop (PPO-2 standard).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.rl.simple.bc_anchor import BCAnchor
from src.rl.simple.policy import BasePolicy
from src.rl.simple.rollout_buffer import RolloutBuffer


def _linear_decay(it: int, warmup: int, start: float, end: float) -> float:
    if warmup <= 0:
        return end
    t = max(0.0, min(1.0, it / float(warmup)))
    return start + (end - start) * t


class PPOTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        optim: torch.optim.Optimizer,
        cfg,
        bc_anchor: Optional[BCAnchor] = None,
    ):
        self.policy = policy
        self.optim = optim
        self.cfg = cfg
        self.bc = bc_anchor
        self._iter = 0

    def _current_bc_coef(self) -> float:
        return _linear_decay(
            self._iter,
            self.cfg.bc_coef_warmup_iters,
            self.cfg.bc_coef_start,
            self.cfg.bc_coef_end,
        )

    def update(self, buffer: RolloutBuffer) -> dict:
        cfg = self.cfg
        metrics = {
            "ppo/pg_loss": 0.0,
            "ppo/value_loss": 0.0,
            "ppo/entropy": 0.0,
            "ppo/bc_loss": 0.0,
            "ppo/approx_kl": 0.0,
            "ppo/clipfrac": 0.0,
            "ppo/grad_norm": 0.0,        # legacy: sum of policy+value pre-clip norms
            "ppo/grad_norm_policy": 0.0,
            "ppo/grad_norm_value": 0.0,
            "ppo/n_minibatches": 0,
            "ppo/early_stopped_at_epoch": -1,
        }
        bc_coef = self._current_bc_coef()
        metrics["ppo/bc_coef"] = bc_coef

        early_stopped_epoch = -1
        early_stop_now = False
        n_mb = 0
        # Track an EMA of approx_kl across minibatches so a single noisy mb
        # doesn't trigger early stop, but a sustained run does. The threshold
        # is the same 1.5×target_kl that the per-epoch check uses.
        kl_ema = 0.0
        kl_ema_alpha = 0.5
        for epoch in range(cfg.update_epochs):
            kl_sum = 0.0
            kl_count = 0
            for mb in buffer.iter_minibatches(cfg.minibatch_size, num_epochs=1):
                logp_new, value_new, entropy = self.policy.evaluate(
                    mb["states"], mb["base_actions"], mb["actions"]
                )
                logp_old = mb["logp_old"]
                value_old = mb["value_old"]
                adv = mb["advantages"]
                ret = mb["returns"]

                # ----- policy loss -----
                ratio = torch.exp(logp_new - logp_old)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv
                pg_loss = -torch.min(surr1, surr2).mean()

                # ----- value loss (clipped) -----
                if cfg.value_clip > 0:
                    v_clipped = value_old + (value_new - value_old).clamp(
                        -cfg.value_clip, cfg.value_clip
                    )
                    v_loss1 = (value_new - ret).pow(2)
                    v_loss2 = (v_clipped - ret).pow(2)
                    value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
                else:
                    value_loss = 0.5 * (value_new - ret).pow(2).mean()

                # ----- entropy bonus -----
                ent_mean = entropy.mean()

                # ----- BC anchor -----
                if self.bc is not None and bc_coef > 0.0:
                    bc_loss = self.bc.loss(self.policy)
                else:
                    bc_loss = torch.zeros((), device=adv.device)

                loss = (
                    pg_loss
                    + cfg.value_coef * value_loss
                    - cfg.ent_coef * ent_mean
                    + bc_coef * bc_loss
                )

                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                # Split grad clip: value head's huge loss (returns ~O(50) →
                # value_loss ~O(1000)) would otherwise dominate a shared norm
                # pool and pollute the policy update. Clip each group on its
                # own budget. See docs/dryrun_jax_step8f_crossanalysis.md §6.
                gn_policy = nn.utils.clip_grad_norm_(
                    self.policy.policy_params(), cfg.max_grad_norm
                )
                gn_value = nn.utils.clip_grad_norm_(
                    self.policy.value_params(), cfg.max_grad_norm
                )
                grad_norm = float(gn_policy) + float(gn_value)
                self.optim.step()

                with torch.no_grad():
                    approx_kl = (logp_old - logp_new).mean().item()
                    clipfrac = (
                        (ratio - 1.0).abs() > cfg.clip_ratio
                    ).float().mean().item()

                metrics["ppo/pg_loss"] += pg_loss.item()
                metrics["ppo/value_loss"] += value_loss.item()
                metrics["ppo/entropy"] += ent_mean.item()
                metrics["ppo/bc_loss"] += float(bc_loss.item())
                metrics["ppo/approx_kl"] += approx_kl
                metrics["ppo/clipfrac"] += clipfrac
                metrics["ppo/grad_norm"] += float(grad_norm)
                metrics["ppo/grad_norm_policy"] += float(gn_policy)
                metrics["ppo/grad_norm_value"] += float(gn_value)
                n_mb += 1
                kl_sum += approx_kl
                kl_count += 1
                kl_ema = (
                    approx_kl if n_mb == 1
                    else kl_ema_alpha * approx_kl + (1 - kl_ema_alpha) * kl_ema
                )
                if cfg.target_kl > 0 and kl_ema > 1.5 * cfg.target_kl:
                    early_stop_now = True
                    early_stopped_epoch = epoch
                    break

            if early_stop_now:
                break
            mean_kl = kl_sum / max(1, kl_count)
            if cfg.target_kl > 0 and mean_kl > 1.5 * cfg.target_kl:
                early_stopped_epoch = epoch
                break

        if n_mb > 0:
            for k in (
                "ppo/pg_loss",
                "ppo/value_loss",
                "ppo/entropy",
                "ppo/bc_loss",
                "ppo/approx_kl",
                "ppo/clipfrac",
                "ppo/grad_norm",
                "ppo/grad_norm_policy",
                "ppo/grad_norm_value",
            ):
                metrics[k] /= n_mb
        metrics["ppo/n_minibatches"] = n_mb
        metrics["ppo/early_stopped_at_epoch"] = early_stopped_epoch
        self._iter += 1
        return metrics
