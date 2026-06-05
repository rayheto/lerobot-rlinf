from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from .result import DiagnosticResult


@dataclass
class DiagnosticContext:
    ref_root: Path
    cand_root: Path
    ref_info: dict[str, Any]
    cand_info: dict[str, Any]


class Diagnostic(ABC):
    name: ClassVar[str] = ""
    category: ClassVar[str] = ""
    thresholds: ClassVar[dict[str, float]] = {}

    @classmethod
    def required_features(cls) -> set[str]:
        return set()

    @abstractmethod
    def run(self, ctx: DiagnosticContext) -> DiagnosticResult: ...

    @classmethod
    def report_template(cls) -> str:
        return ""
