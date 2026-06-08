"""Patch DSRL's target-shadow init/update to only touch *trainable* params.

RLinf's EmbodiedSACFSDPPolicy._init_target_shadow unconditionally fp32-clones
every parameter of the target model. For pi0.5 (~3.6B params, bf16) that costs
+14 GB of GPU just for a shadow buffer of frozen weights that never EMA-update.

We replace it (and the matching soft_update) to only shadow params whose
*online* counterpart has requires_grad=True (i.e. the ~500K SAC head +
encoders). Frozen backbone params are skipped: target stays bitwise-equal to
online (both never change), no EMA needed.

Saves ~14 GB on single-GPU runs. Idempotent. Activated via
RLINF_DSRL_SHADOW_PATCH=1 (default on); set =0 to disable.
"""
from __future__ import annotations

import importlib
import os
from typing import Optional

_PATCHED = False
_HOOKED = False
_FSDP_PATCHED = False


def _patch_fsdp_writeback() -> None:
    """Tolerate stale grad shape in FSDP writeback after offload roundtrip.

    With use_orig_params=True + enable_offload (custom CPU↔GPU shuffle),
    FSDP's _writeback_tensor sees grads whose shape == orig param shape
    (e.g. [128,128]) instead of the flattened expected_shape ([16384]),
    and raises RuntimeError. numel matches — reshape and proceed.
    Reproduced at ~10 steps on prod DSRL pi0.5 + FSDP no_shard."""
    global _FSDP_PATCHED
    if _FSDP_PATCHED:
        return
    try:
        from torch.distributed.fsdp import _flat_param as _fp
    except Exception:
        return
    orig = _fp.FlatParamHandle._writeback_tensor

    def _patched(self, src_tensor, dst_tensor, tensor_index, expected_shape, offset, is_param):
        if (
            src_tensor is not None
            and src_tensor.shape != expected_shape
            and src_tensor.numel() == expected_shape.numel()
        ):
            src_tensor = src_tensor.reshape(expected_shape)
        return orig(self, src_tensor, dst_tensor, tensor_index, expected_shape, offset, is_param)

    _fp.FlatParamHandle._writeback_tensor = _patched
    _FSDP_PATCHED = True

_TARGET_MODULE = "rlinf.workers.actor.fsdp_sac_policy_worker"


def _apply(mod) -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    Cls = getattr(mod, "EmbodiedSACFSDPPolicy", None)
    if Cls is None:
        return False

    import torch

    def _init_target_shadow(self):
        """Only shadow trainable params (saves ~14 GB on frozen pi0.5)."""
        self._target_shadow_f32 = {}
        trainable_names = {
            n for n, p in self.model.named_parameters() if p.requires_grad
        }
        # Note: target may be missing entries (e.g. paligemma_with_expert
        # stripped from target_module to save memory). Iterate target's params
        # and only shadow those that also exist as trainable in main.
        for name, param in self.target_model.named_parameters():
            if name in trainable_names:
                self._target_shadow_f32[name] = param.data.float().clone()

    def soft_update_target_model(self, tau: Optional[float] = None):
        """Look up target params by name. Target may have fewer params than
        main (frozen backbone stripped from target_module to save memory).
        Skip any trainable param of main that has no counterpart in target."""
        if tau is None:
            tau = self.cfg.algorithm.tau
        assert self.target_model_initialized
        shadow = getattr(self, "_target_shadow_f32", None)
        target_params = dict(self.target_model.named_parameters())
        with torch.no_grad():
            for name, online_param in self.model.named_parameters():
                if not online_param.requires_grad:
                    continue
                target_param = target_params.get(name)
                if target_param is None:
                    continue
                if shadow is not None and name in shadow:
                    s = shadow[name]
                    if "q_head" in name or self.target_update_type == "all":
                        s.mul_(1.0 - tau).add_(online_param.data.float(), alpha=tau)
                    else:
                        s.copy_(online_param.data.float())
                    target_param.data.copy_(s.to(target_param.data.dtype))
                else:
                    if "q_head" in name or self.target_update_type == "all":
                        target_param.data.mul_(1.0 - tau)
                        target_param.data.add_(online_param.data * tau)
                    else:
                        target_param.data.copy_(online_param.data)

    Cls._init_target_shadow = _init_target_shadow
    Cls.soft_update_target_model = soft_update_target_model
    _PATCHED = True
    return True


def patch() -> bool:
    """Install a lazy import hook; patches only when the SAC worker module is
    actually imported (i.e. in the actor Ray process). Cheap no-op elsewhere."""
    global _HOOKED
    if _HOOKED:
        return True
    if os.environ.get("RLINF_DSRL_SHADOW_PATCH", "1") != "1":
        return False

    import sys
    if _TARGET_MODULE in sys.modules:
        return _apply(sys.modules[_TARGET_MODULE])

    import builtins
    _orig_import = builtins.__import__

    def _hooked_import(name, globals=None, locals=None, fromlist=(), level=0):
        m = _orig_import(name, globals, locals, fromlist, level)
        if not _PATCHED and _TARGET_MODULE in sys.modules:
            try:
                _apply(sys.modules[_TARGET_MODULE])
            except Exception:
                pass
        return m

    builtins.__import__ = _hooked_import
    _HOOKED = True
    return True
