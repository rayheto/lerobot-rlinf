"""Inject SO-101 pick_orange openpi dataconfig + policy into upstream RLinf.

third_party/RLinf is pinned to upstream main (clean, no fork patches). The
SO-101 dataconfig + policy live in this repo (src/rl/openpi_so101_*.py) and
get grafted into the rlinf.models.embodiment.openpi namespace at import time,
so that:
  - `from rlinf.models.embodiment.openpi.policies import so101_lift_policy` works
  - `get_openpi_config("pi05_isaaclab_so101_pick_orange")` returns our TrainConfig

Activated via RLINF_OPENPI_SO101_PATCH=1 (default on; see sitecustomize.py).
Idempotent.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import pathlib
import sys

_PATCHED = False
_HOOKED = False
_DATACONFIG_MOD = "rlinf.models.embodiment.openpi.dataconfig"
_POLICIES_MOD = "rlinf.models.embodiment.openpi.policies"

_HERE = pathlib.Path(__file__).resolve().parent
_POLICY_SRC = _HERE / "openpi_so101_policy.py"
_DATACONFIG_SRC = _HERE / "openpi_so101_dataconfig.py"


def _load_file_as(module_name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _apply() -> bool:
    global _PATCHED
    if _PATCHED:
        return True

    policies_pkg = sys.modules.get(_POLICIES_MOD)
    dataconfig_pkg = sys.modules.get(_DATACONFIG_MOD)
    if policies_pkg is None or dataconfig_pkg is None:
        return False

    # 1) Graft policy module as a submodule of rlinf.models...openpi.policies.
    #    The dataconfig file does `from rlinf...policies import so101_lift_policy`,
    #    which resolves via sys.modules.
    policy_full = f"{_POLICIES_MOD}.so101_lift_policy"
    if policy_full not in sys.modules:
        policy_mod = _load_file_as(policy_full, _POLICY_SRC)
        setattr(policies_pkg, "so101_lift_policy", policy_mod)

    # 2) Graft dataconfig module similarly.
    dc_full = f"{_DATACONFIG_MOD}.isaaclab_so101_dataconfig"
    if dc_full not in sys.modules:
        dc_mod = _load_file_as(dc_full, _DATACONFIG_SRC)
        setattr(dataconfig_pkg, "isaaclab_so101_dataconfig", dc_mod)
    else:
        dc_mod = sys.modules[dc_full]

    # 3) Append the SO-101 pick_orange TrainConfig to upstream's _CONFIGS_DICT.
    #    Mirror the registration that lived in our pre-reset fork's __init__.py.
    import openpi.models.pi0_config as pi0_config
    import openpi.training.weight_loaders as weight_loaders
    from openpi.training.config import AssetsConfig, DataConfig, TrainConfig

    cfg = TrainConfig(
        name="pi05_isaaclab_so101_pick_orange",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=50, discrete_state_input=False
        ),
        data=dc_mod.LeRobotIsaacLabSo101PickOrangeDataConfig(
            repo_id="LightwheelAI/leisaac-pick-orange",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="checkpoints/torch/pi05_isaaclab_so101_pick_orange/assets"
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi05_base/params"
        ),
        pytorch_weight_path="checkpoints/torch/pi05_base",
        num_train_steps=30_000,
    )
    dataconfig_pkg._CONFIGS_DICT[cfg.name] = cfg
    if cfg not in dataconfig_pkg._CONFIGS:
        dataconfig_pkg._CONFIGS.append(cfg)

    _PATCHED = True
    return True


def patch() -> bool:
    """Lazy import hook: graft SO-101 onto rlinf.models...openpi when that
    package is imported. Cheap no-op outside RLinf processes."""
    global _HOOKED
    if _HOOKED:
        return True
    if os.environ.get("RLINF_OPENPI_SO101_PATCH", "1") != "1":
        return False

    # If both target modules are already loaded, apply now.
    if _DATACONFIG_MOD in sys.modules and _POLICIES_MOD in sys.modules:
        return _apply()

    import builtins
    _orig_import = builtins.__import__

    def _hooked_import(name, globals=None, locals=None, fromlist=(), level=0):
        m = _orig_import(name, globals, locals, fromlist, level)
        if not _PATCHED:
            if _DATACONFIG_MOD in sys.modules and _POLICIES_MOD in sys.modules:
                try:
                    _apply()
                except Exception:
                    pass
        return m

    builtins.__import__ = _hooked_import
    _HOOKED = True
    return True
