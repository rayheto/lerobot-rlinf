"""RLinf extension hook for project-local residual PPO components."""
from __future__ import annotations

from src.rl.rlinf_residual.env import register_env
from src.rl.rlinf_residual.model import register_model


def register() -> None:
    register_model()
    register_env()
