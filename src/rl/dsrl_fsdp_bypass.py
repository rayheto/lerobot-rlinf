"""Bypass FSDP wrapping entirely on single-GPU + no_shard runs.

On a single 4090 with sharding_strategy=no_shard and world_size=1, FSDP
has no real job — it doesn't shard, doesn't all-gather, doesn't reduce.
But its flat_param/use_orig_params machinery + RLinf's custom CPU↔GPU
offload still interact in ways that corrupt FSDP-internal state:

  - flat_param.grad shape (1D flat) drifts from orig param.grad shape (2D)
    after offload → `Cannot writeback when the gradient shape changes`
  - flat_param.grad device drifts (CPU) from flat_param.data (GPU) after
    onload → `attempting to assign a gradient with device type 'cpu' to
    a tensor with device type 'cuda'`

Since FSDP is providing no value here, the cleanest fix is to skip it:
when RLINF_FSDP_BYPASS=1, monkey-patch FSDPStrategy so that wrap_model
just returns model.to(device), and offload/onload/clip_grad walk plain
named_parameters() instead of FSDP handles. State_dict goes through
nn.Module.state_dict() — which is what patch_syncer already expects.

Activated via RLINF_FSDP_BYPASS=1 (default off).
"""
from __future__ import annotations

import os
from contextlib import nullcontext

_PATCHED = False
_HOOKED = False
_STRATEGY_MODULE = "rlinf.hybrid_engines.fsdp.strategy.fsdp"
_BYPASS_ATTR = "_rlinf_bypass_no_fsdp"


def _apply(mod) -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    Cls = getattr(mod, "FSDPStrategy", None)
    if Cls is None:
        return False

    import torch

    def _is_bypass(self, model) -> bool:
        return getattr(model, _BYPASS_ATTR, False)

    _orig_wrap_model = Cls.wrap_model

    def wrap_model(self, model, device_mesh):
        # Only bypass when both flag is set AND the run is genuinely
        # single-GPU + no_shard (the only case FSDP is decorative).
        if (
            os.environ.get("RLINF_FSDP_BYPASS", "0") == "1"
            and self.cfg.fsdp_config.sharding_strategy == "no_shard"
            and int(os.environ.get("WORLD_SIZE", "1")) == 1
        ):
            device = torch.device(
                f"cuda:{os.environ.get('LOCAL_RANK', '0')}"
            )
            model = model.to(device)
            setattr(model, _BYPASS_ATTR, True)
            # SAC worker calls self.model.clip_grad_norm_(max_norm=...) —
            # FSDP module ships that method; plain nn.Module doesn't.
            def _clip_grad_norm_(max_norm, norm_type=2.0):
                return torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm, float(norm_type)
                )
            model.clip_grad_norm_ = _clip_grad_norm_
            return model
        return _orig_wrap_model(self, model, device_mesh)

    _orig_offload = Cls.offload_param_and_grad

    @torch.no_grad()
    def offload_param_and_grad(self, model, offload_grad):
        if not _is_bypass(self, model):
            return _orig_offload(self, model, offload_grad)
        for _, p in model.named_parameters():
            p.data = p.data.to("cpu", non_blocking=True)
            if offload_grad and p.grad is not None:
                p.grad = p.grad.to("cpu", non_blocking=True)
        for _, b in model.named_buffers():
            b.data = b.data.to("cpu", non_blocking=True)
        torch.cuda.empty_cache()

    _orig_onload = Cls.onload_param_and_grad

    @torch.no_grad()
    def onload_param_and_grad(self, model, device, onload_grad):
        if not _is_bypass(self, model):
            return _orig_onload(self, model, device, onload_grad)
        for _, p in model.named_parameters():
            p.data = p.data.to(device, non_blocking=True)
            if onload_grad and p.grad is not None:
                p.grad = p.grad.to(device, non_blocking=True)
        for _, b in model.named_buffers():
            b.data = b.data.to(device, non_blocking=True)
        torch.cuda.empty_cache()

    _orig_clip = Cls.clip_grad_norm_

    def clip_grad_norm_(self, model, norm_type=2.0):
        if not _is_bypass(self, model):
            return _orig_clip(self, model, norm_type)
        max_norm = float(self.cfg.optim.clip_grad)
        return (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm, float(norm_type)
            )
            .cpu()
            .item()
        )

    _orig_get_sd = Cls.get_model_state_dict

    def get_model_state_dict(self, model, cpu_offload, full_state_dict):
        if not _is_bypass(self, model):
            return _orig_get_sd(self, model, cpu_offload, full_state_dict)
        sd = model.state_dict()
        if cpu_offload:
            sd = {k: v.detach().to("cpu") for k, v in sd.items()}
        return sd

    _orig_get_opt_sd = Cls.get_optimizer_state_dict

    def get_optimizer_state_dict(self, model, optimizer):
        if not _is_bypass(self, model):
            return _orig_get_opt_sd(self, model, optimizer)
        return optimizer.state_dict()

    _orig_before_mb = Cls.before_micro_batch

    def before_micro_batch(self, model, is_last_micro_batch):
        if not _is_bypass(self, model):
            return _orig_before_mb(self, model, is_last_micro_batch)
        return nullcontext()

    Cls.wrap_model = wrap_model
    Cls.offload_param_and_grad = offload_param_and_grad
    Cls.onload_param_and_grad = onload_param_and_grad
    Cls.clip_grad_norm_ = clip_grad_norm_
    Cls.get_model_state_dict = get_model_state_dict
    Cls.get_optimizer_state_dict = get_optimizer_state_dict
    Cls.before_micro_batch = before_micro_batch
    _PATCHED = True
    return True


def patch() -> bool:
    """Lazy-import hook; only patches FSDPStrategy when it's imported
    in the Ray actor subprocess. Cheap no-op elsewhere."""
    global _HOOKED
    if _HOOKED:
        return True
    if os.environ.get("RLINF_FSDP_BYPASS", "0") != "1":
        return False

    import sys
    if _STRATEGY_MODULE in sys.modules:
        return _apply(sys.modules[_STRATEGY_MODULE])

    import builtins
    _orig_import = builtins.__import__

    def _hooked_import(name, globals=None, locals=None, fromlist=(), level=0):
        m = _orig_import(name, globals, locals, fromlist, level)
        if not _PATCHED and _STRATEGY_MODULE in sys.modules:
            try:
                _apply(sys.modules[_STRATEGY_MODULE])
            except Exception:
                pass
        return m

    builtins.__import__ = _hooked_import
    _HOOKED = True
    return True
