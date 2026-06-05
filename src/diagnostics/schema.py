from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Mismatch:
    feature: str
    field: str
    ref: Any
    cand: Any

    def __str__(self) -> str:
        return f"{self.feature}.{self.field}: ref={self.ref!r} cand={self.cand!r}"


def validate_pair(
    ref_info: dict[str, Any],
    cand_info: dict[str, Any],
    required_features: set[str],
) -> list[Mismatch]:
    mismatches: list[Mismatch] = []
    ref_feats = ref_info.get("features", {})
    cand_feats = cand_info.get("features", {})

    for feat in sorted(required_features):
        if feat not in ref_feats:
            mismatches.append(Mismatch(feat, "presence(ref)", True, False))
            continue
        if feat not in cand_feats:
            mismatches.append(Mismatch(feat, "presence(cand)", True, False))
            continue
        rf, cf = ref_feats[feat], cand_feats[feat]
        if rf.get("dtype") != cf.get("dtype"):
            mismatches.append(Mismatch(feat, "dtype", rf.get("dtype"), cf.get("dtype")))
        if list(rf.get("shape", [])) != list(cf.get("shape", [])):
            mismatches.append(Mismatch(feat, "shape", rf.get("shape"), cf.get("shape")))
    return mismatches
