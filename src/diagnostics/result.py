from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Status(str, Enum):
    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


@dataclass
class DiagnosticResult:
    name: str
    category: str
    status: Status
    metrics: dict[str, float] = field(default_factory=dict)
    narrative: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "category": self.category,
            "status": self.status.value,
            "metrics": dict(self.metrics),
            "narrative": list(self.narrative),
        }
        if self.error is not None:
            d["error"] = self.error
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DiagnosticResult":
        return cls(
            name=d["name"],
            category=d["category"],
            status=Status(d["status"]),
            metrics=dict(d.get("metrics", {})),
            narrative=list(d.get("narrative", [])),
            error=d.get("error"),
        )
