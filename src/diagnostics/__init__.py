from __future__ import annotations

import traceback
from typing import Iterable

from .base import Diagnostic, DiagnosticContext
from .registry import REGISTRY, register_diagnostic
from .result import DiagnosticResult, Status

from . import modules as _modules  # noqa: F401  triggers registration


def run_all(ctx: DiagnosticContext, names: Iterable[str] | None = None) -> list[DiagnosticResult]:
    selected = list(names) if names is not None else sorted(REGISTRY)
    out: list[DiagnosticResult] = []
    for n in selected:
        cls = REGISTRY.get(n)
        if cls is None:
            out.append(
                DiagnosticResult(
                    name=n, category="unknown", status=Status.ERROR,
                    error=f"diagnostic '{n}' not registered",
                )
            )
            continue
        try:
            out.append(cls().run(ctx))
        except Exception as exc:
            out.append(
                DiagnosticResult(
                    name=cls.name, category=cls.category, status=Status.ERROR,
                    error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                )
            )
    return out


__all__ = [
    "Diagnostic", "DiagnosticContext", "DiagnosticResult", "Status",
    "REGISTRY", "register_diagnostic", "run_all",
]
