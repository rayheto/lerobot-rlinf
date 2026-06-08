"""Runtime injection of the SO-101 PickOrange env into RLinf's registry.

RLinf's isaaclab registry is a plain dict (rlinf.envs.isaaclab.REGISTER_ISAACLAB_ENVS),
keyed by the gym env id string in the config. We add our task id without
modifying any file under third_party/.
"""
from __future__ import annotations

import importlib

_TASK_ID = "LeIsaac-SO101-PickOrange-v0"


def patch() -> None:
    """Idempotent monkey-patch. Safe to call multiple times."""
    rlinf_isaaclab = importlib.import_module("rlinf.envs.isaaclab")
    registry = rlinf_isaaclab.REGISTER_ISAACLAB_ENVS

    if _TASK_ID in registry:
        return

    from src.rl.envs.isaaclab_pick_orange import IsaaclabPickOrangeEnv

    registry[_TASK_ID] = IsaaclabPickOrangeEnv


def is_patched() -> bool:
    try:
        rlinf_isaaclab = importlib.import_module("rlinf.envs.isaaclab")
    except ImportError:
        return False
    return _TASK_ID in rlinf_isaaclab.REGISTER_ISAACLAB_ENVS
