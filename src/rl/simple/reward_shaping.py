"""Post-hoc reward shaping on top of the env's sparse 3-stage reward.

Four terms layered on top of the env reward:

  ood_penalty    = -ood_coef * d_norm    via OodKNNPenalty (EXP_05 = 5.07)
  survival_cost  = const < 0 per step    while not done    (EXP_03 = 2.735)
  dense_eo       = -alpha * max(0, d_ee_orange - floor)    Plan B (precision push)
  dense_lift     = +beta  * max(0, lift_dz)                Plan B (grasp confirm)

dense_eo / dense_lift target the JAX-24999 dryrun finding: the SFT policy
parks ee ~7 cm from the orange and never enters the 5 cm grasp gate, so the
sparse predicate never fires. See docs/dryrun_jax_24999_diagnostic.md.

Reward additivity stays intact: the env reward (sparse stages: grasp/carry/
place/drop/timeout) is NOT modified inside the env. Only added here.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from src.rl.envs.ood_kdtree import OodKNNConfig, OodKNNPenalty


class ShapedReward:
    def __init__(self, cfg):
        self._ood = OodKNNPenalty.get(
            OodKNNConfig(
                demo_dataset_path=cfg.demo_dataset_path,
                k_neighbors=cfg.ood_k_neighbors,
                coef=cfg.ood_coef,
            )
        )
        self._survival = float(cfg.survival_cost)
        self._dense_eo_coef = float(getattr(cfg, "dense_eo_coef", 0.0))
        self._dense_eo_floor = float(getattr(cfg, "dense_eo_floor", 0.05))
        self._dense_lift_coef = float(getattr(cfg, "dense_lift_coef", 0.0))

    def __call__(
        self,
        env_reward: torch.Tensor,
        joint_pos: torch.Tensor,
        done_or_trunc: torch.Tensor,
        aux: Optional[Dict[str, torch.Tensor]] = None,
        orange_init_z: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Returns (shaped_reward [Nenv], info_dict).

        joint_pos: (Nenv, state_dim) — the OBSERVATION at step start (pre-step).
        done_or_trunc: (Nenv,) bool — survival cost is zeroed on terminal steps.
        aux: dict from env._last_aux (POST-step state). Must contain
             "ee_pos" (Nenv,3) and "orange001_pos" (Nenv,3) for dense_eo /
             dense_lift to fire. None → dense terms = 0 (back-compat).
        orange_init_z: (Nenv,) — env._orange_init_z. Required if aux is given.
        """
        ood = self._ood.reward(joint_pos)  # (Nenv,) already negative, GPU
        ood = ood.to(env_reward.device, env_reward.dtype)
        survive_mask = (~done_or_trunc.bool()).to(env_reward.dtype)
        survival = torch.full_like(env_reward, self._survival) * survive_mask

        dense_eo = torch.zeros_like(env_reward)
        dense_lift = torch.zeros_like(env_reward)
        if aux is not None and "ee_pos" in aux and "orange001_pos" in aux:
            ee = aux["ee_pos"].to(env_reward.device, env_reward.dtype)
            orange = aux["orange001_pos"].to(env_reward.device, env_reward.dtype)
            d_eo = torch.linalg.vector_norm(ee - orange, dim=-1)  # (Nenv,)
            if self._dense_eo_coef > 0.0:
                excess = torch.clamp(d_eo - self._dense_eo_floor, min=0.0)
                dense_eo = -self._dense_eo_coef * excess
            if self._dense_lift_coef > 0.0 and orange_init_z is not None:
                # orange001_pos is (Nenv,3); env._orange_init_z is (Nenv,3) — z
                # for all 3 oranges. We only shape orange001 lift, so take col 0.
                z = orange[..., 2]                              # (Nenv,)
                z0_full = orange_init_z.to(z.device, z.dtype)   # (Nenv,3)
                z0 = z0_full[..., 0] if z0_full.ndim == 2 else z0_full
                lift = torch.clamp(z - z0, min=0.0)
                dense_lift = self._dense_lift_coef * lift

        shaped = env_reward + ood + survival + dense_eo + dense_lift
        info = {
            "mean_env_reward": env_reward.mean().item(),
            "mean_ood_penalty": ood.mean().item(),
            "mean_survival_cost": survival.mean().item(),
            "mean_dense_eo": dense_eo.mean().item(),
            "mean_dense_lift": dense_lift.mean().item(),
            "mean_shaped_reward": shaped.mean().item(),
        }
        return shaped, info
