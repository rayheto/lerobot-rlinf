"""Lazy AdamW -> bnb.AdamW8bit shim.

Why: RLinf's fsdp_model_manager.build_optimizer hard-codes
``torch.optim.AdamW(...)`` at line 533. On a single 4090, pi05's ~693M
trainable params need ~8.3GB of fp32 AdamW state which OOMs the GPU.
bitsandbytes 8-bit AdamW cuts that to ~1.4GB.

This module installs a *lazy* shim: ``torch.optim.AdamW`` becomes a factory
that imports bitsandbytes only when called. Eager-importing bnb in
sitecustomize blew up CPU RAM — Ray spawns ~32 helper procs (raylet,
log-monitor, dashboard, idle workers, …) on a 32-core box, none of which
need bnb, but each was paying its ~400MB import cost.

Gated by RLINF_BNB_ADAMW8BIT=1.
"""
from __future__ import annotations

import os


_PATCHED = False


def _make_lazy_adamw8bit():
    """Return a callable that builds bnb.optim.AdamW8bit on first invocation."""
    def _factory(*args, **kwargs):
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(*args, **kwargs)
    _factory.__name__ = "LazyAdamW8bit"
    return _factory


def patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    if os.environ.get("RLINF_BNB_ADAMW8BIT", "0") != "1":
        return False

    import torch.optim
    torch.optim.AdamW = _make_lazy_adamw8bit()
    _PATCHED = True
    return True


def is_patched() -> bool:
    return _PATCHED
