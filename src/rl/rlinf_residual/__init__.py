"""RLinf glue for SO-101 residual PPO."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[3]
for _p in (
    _REPO,
    _REPO / "third_party" / "RLinf",
    _REPO / "third_party" / "openpi" / "src",
    _REPO / "third_party" / "openpi" / "packages" / "openpi-client" / "src",
):
    _sp = str(_p)
    if _sp not in _sys.path:
        _sys.path.insert(0, _sp)
