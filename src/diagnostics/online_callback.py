"""Programmatic diagnostics runner for the RL eval loop.

Wraps `src.diagnostics.run_all` so the RL eval orchestrator can run EXP_01 /
EXP_03 / EXP_05 against a (reference, candidate) LeRobot v2.1 dataset pair and
get a flat `{exp/metric: float}` dict back, without going through the CLI.

The tensorboard write itself stays in `src/rl/eval_with_diagnostics.py` so
this module has zero RLinf / torch dependencies and can be reused by other
callers (offline reporting, CI checks).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from . import DiagnosticContext, run_all
from .io import load_info
from .result import DiagnosticResult, Status

_STATUS_CODE: dict[Status, float] = {
    Status.OK: 0.0,
    Status.WARNING: 1.0,
    Status.CRITICAL: 2.0,
    Status.SKIPPED: -1.0,
    Status.ERROR: -2.0,
}

# Default subset aligned with sft_diagnostics_findings.md — what we want to
# watch on tensorboard during PPO. EXP_02 (compounding_error) and EXP_04
# (action_smoothness) are useful but not load-bearing for this task.
DEFAULT_MODULES: tuple[str, ...] = (
    "EXP_01_Mode_Averaging",
    "EXP_03_Episode_Length_Inflation",
    "EXP_05_State_Coverage_Divergence",
)


def run_diagnostics_on_rollout_dir(
    ref_root: Path | str,
    cand_root: Path | str,
    modules: Iterable[str] | None = DEFAULT_MODULES,
) -> tuple[dict[str, float], list[DiagnosticResult]]:
    """Run diagnostics, return flat tensorboard-ready dict plus raw results.

    Returns:
        (flat, results) where flat maps "EXP_XX/metric_name" -> float and
        includes a "EXP_XX/status_code" entry per module
        (OK=0, WARNING=1, CRITICAL=2, SKIPPED=-1, ERROR=-2). The second tuple
        element is the original list of DiagnosticResult for callers that want
        narratives or the markdown report.
    """
    ref_root = Path(ref_root)
    cand_root = Path(cand_root)
    ctx = DiagnosticContext(
        ref_root=ref_root,
        cand_root=cand_root,
        ref_info=load_info(ref_root),
        cand_info=load_info(cand_root),
    )
    names = list(modules) if modules is not None else None
    results = run_all(ctx, names=names)

    flat: dict[str, float] = {}
    for r in results:
        flat[f"{r.name}/status_code"] = _STATUS_CODE.get(r.status, -2.0)
        for k, v in r.metrics.items():
            try:
                flat[f"{r.name}/{k}"] = float(v)
            except (TypeError, ValueError):
                continue
    return flat, results


__all__ = ["run_diagnostics_on_rollout_dir", "DEFAULT_MODULES"]
