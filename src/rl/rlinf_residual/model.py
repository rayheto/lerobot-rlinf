"""Project-side RLinf model registration for SO-101 residual PPO."""
from __future__ import annotations

import torch

MODEL_TYPE = "so101_residual_mlp"


def build_model(cfg, torch_dtype=None):
    from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy

    model = MLPPolicy(
        obs_dim=int(cfg.get("obs_dim", 12)),
        action_dim=int(cfg.get("action_dim", 6)),
        num_action_chunks=int(cfg.get("num_action_chunks", 1)),
        add_value_head=bool(cfg.get("add_value_head", True)),
        add_q_head=bool(cfg.get("add_q_head", False)),
        q_head_type=str(cfg.get("q_head_type", "default")),
    )

    if hasattr(model, "actor_logstd"):
        with torch.no_grad():
            model.actor_logstd.fill_(float(cfg.get("init_log_std", -1.0)))

    if bool(cfg.get("zero_init_actor_mean", True)) and hasattr(model, "actor_mean"):
        with torch.no_grad():
            model.actor_mean.weight.zero_()
            if model.actor_mean.bias is not None:
                model.actor_mean.bias.zero_()

    if bool(cfg.get("zero_init_value_head", True)) and hasattr(model, "value_head"):
        last = getattr(model.value_head, "mlp", [None])[-1]
        if isinstance(last, torch.nn.Linear):
            with torch.no_grad():
                last.weight.zero_()
                if last.bias is not None:
                    last.bias.zero_()

    if torch_dtype is not None:
        model = model.to(dtype=torch_dtype)
    return model


def register_model() -> None:
    from rlinf.models import register_model as _register_model

    _register_model(MODEL_TYPE, build_model, category="embodied", force=True)
