"""Simple single-process PPO framework for SO-101 pick_orange post-SFT.

Self-contained replacement for the rlinf PPO scaffolding in src/rl/. Reuses:
  - src/rl/envs/isaaclab_pick_orange.py (IsaaclabPickOrangeEnv + sparse reward)
  - src/rl/envs/ood_kdtree.py (OodKNNPenalty)
  - src/rl/openpi_so101_policy.py + openpi_patch.py (pi05 IO transforms + patch)

Specifically targets the SFT diagnostic findings in docs/sft_diagnostics_findings.md:
  EXP_05 OOD coverage 5.07 (main)  ← OOD penalty term
  EXP_03 length inflation 2.735    ← per-step survival cost
  EXP_01 mode covering 0.726       ← BC anchor to demo dataset
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Path setup: openpi + RLinf live under third_party/ and are not pip-installed.
# Mirror src/rl/run.sh's PYTHONPATH so any importer of this package can use
# `import openpi.*` and `from rlinf.envs... import ...` directly.
# ---------------------------------------------------------------------------
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

