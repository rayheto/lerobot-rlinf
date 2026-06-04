"""Phase 1.5a — audit lerobot SFT ckpt keys vs RLinf openpi expected keys.

Run with the RLinf .venv python:
    /home/hlei/RLinf/.venv/bin/python scripts/audit_ckpt_keys.py

Writes a human-reviewable report to outputs/sft_pi05_sponge/key_audit.txt.
The shape-matched orphan section is the important one — those are the
keys we have to hand-rename in Step C beyond the bulk `model.` prefix strip.
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import safetensors
import torch

LEROBOT_CKPT = Path(os.environ.get(
    "LEROBOT_CKPT",
    "/home/hlei/robotic/lerobot-rlinf/outputs/sft_pi05_sponge/checkpoints/010000/pretrained_model/model.safetensors",
))
REPORT_PATH = Path(os.environ.get(
    "AUDIT_REPORT_PATH",
    "/home/hlei/robotic/lerobot-rlinf/outputs/sft_pi05_sponge/key_audit.txt",
))
OPENPI_CONFIG_NAME = "pi05_isaaclab_so101_lift"


def load_lerobot_shapes() -> dict[str, tuple[int, ...]]:
    out: dict[str, tuple[int, ...]] = {}
    with safetensors.safe_open(str(LEROBOT_CKPT), framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_slice(k)
            out[k] = tuple(t.get_shape())
    return out


def build_openpi_model_on_meta():
    """Instantiate OpenPi0ForRLActionPrediction on the meta device.

    Meta device avoids allocating ~7 GB host RAM just to read parameter
    names + shapes. HF transformers respects torch.device("meta") for
    nn.Linear / nn.Embedding etc., so this is safe for an audit.
    """
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.openpi_action_model import (
        OpenPi0Config,
        OpenPi0ForRLActionPrediction,
    )

    train_config = get_openpi_config(
        OPENPI_CONFIG_NAME,
        model_path="/tmp/__nonexistent_for_audit__",
    )
    model_config = OpenPi0Config(**train_config.model.__dict__)

    with torch.device("meta"):
        model = OpenPi0ForRLActionPrediction(model_config)
    return model


def load_openpi_shapes() -> dict[str, tuple[int, ...]]:
    model = build_openpi_model_on_meta()
    out: dict[str, tuple[int, ...]] = {}
    for name, p in model.named_parameters():
        out[name] = tuple(p.shape)
    for name, b in model.named_buffers():
        if name not in out:
            out[name] = tuple(b.shape)
    return out


def _emit(lines: list[str], s: str = "") -> None:
    print(s)
    lines.append(s)


def main() -> int:
    report: list[str] = []

    print(f"[audit] loading lerobot shapes from {LEROBOT_CKPT}")
    lerobot = load_lerobot_shapes()
    print(f"[audit] lerobot: {len(lerobot)} tensors")

    print(f"[audit] building meta-device openpi model ({OPENPI_CONFIG_NAME})")
    openpi = load_openpi_shapes()
    print(f"[audit] openpi:  {len(openpi)} tensors")

    # Round 1: strip `model.` prefix and match.
    renamed: dict[str, tuple[str, tuple[int, ...]]] = {}
    for k, shape in lerobot.items():
        nk = k.removeprefix("model.")
        renamed[nk] = (k, shape)

    matched: list[tuple[str, str]] = []  # (lerobot_key, openpi_key)
    shape_mismatch: list[tuple[str, str, tuple, tuple]] = []
    lerobot_only: dict[str, tuple[int, ...]] = {}
    for nk, (lk, lshape) in renamed.items():
        if nk in openpi:
            if openpi[nk] == lshape:
                matched.append((lk, nk))
            else:
                shape_mismatch.append((lk, nk, lshape, openpi[nk]))
        else:
            lerobot_only[lk] = lshape

    openpi_only: dict[str, tuple[int, ...]] = {
        k: shape for k, shape in openpi.items() if k not in renamed
    }

    _emit(report, "=== Summary ===")
    _emit(report, f"lerobot total: {len(lerobot)}")
    _emit(report, f"openpi total:  {len(openpi)}")
    _emit(report, f"matched after `model.` strip: {len(matched)}")
    _emit(report, f"shape mismatch on matched name: {len(shape_mismatch)}")
    _emit(report, f"lerobot_only (no openpi target): {len(lerobot_only)}")
    _emit(report, f"openpi_only  (no lerobot source): {len(openpi_only)}")
    _emit(report)

    # Spot-check 10 matched keys for shape parity (sanity).
    if matched:
        rng = random.Random(0)
        sample = rng.sample(matched, min(10, len(matched)))
        _emit(report, "=== Spot-check (10 matched) ===")
        for lk, ok in sample:
            _emit(report, f"  {lk}  ==  {ok}  shape={lerobot[lk]}")
        _emit(report)

    # Shape mismatches first — these would silently fail under strict=False.
    if shape_mismatch:
        _emit(report, "=== !!! Shape mismatch on matched names !!! ===")
        for lk, ok, ls, os_ in shape_mismatch:
            _emit(report, f"  {lk} -> {ok}: lerobot {ls} vs openpi {os_}")
        _emit(report)

    # Round 2: shape-bucketed orphan pairing — the focal point per user's note.
    by_shape_lerobot: dict[tuple, list[str]] = defaultdict(list)
    by_shape_openpi: dict[tuple, list[str]] = defaultdict(list)
    for k, s in lerobot_only.items():
        by_shape_lerobot[s].append(k)
    for k, s in openpi_only.items():
        by_shape_openpi[s].append(k)

    all_shapes = sorted(
        set(by_shape_lerobot) | set(by_shape_openpi),
        key=lambda s: (len(s), s),
    )

    _emit(report, "=== Shape-matched orphan candidates (manual review) ===")
    _emit(report, "Both-sides-have-orphans-at-this-shape (likely renames):")
    has_pair = False
    for s in all_shapes:
        lro = by_shape_lerobot.get(s, [])
        oro = by_shape_openpi.get(s, [])
        if lro and oro:
            has_pair = True
            _emit(report, f"\n  shape {s}:")
            _emit(report, f"    lerobot_only ({len(lro)}): {lro}")
            _emit(report, f"    openpi_only  ({len(oro)}): {oro}")
            if len(lro) == 1 and len(oro) == 1:
                _emit(report, f"    → unambiguous rename: {lro[0]} -> {oro[0]}")
    if not has_pair:
        _emit(report, "  (none — all orphans are one-sided)")
    _emit(report)

    _emit(report, "=== Pure orphans (no shape match on the other side) ===")
    _emit(report, "lerobot_only with no openpi counterpart at that shape (will be dropped):")
    for s in sorted(by_shape_lerobot, key=lambda s: (len(s), s)):
        if not by_shape_openpi.get(s):
            for k in by_shape_lerobot[s]:
                _emit(report, f"  {k}  shape={s}")
    _emit(report)
    _emit(report, "openpi_only with no lerobot counterpart at that shape (random init under strict=False):")
    for s in sorted(by_shape_openpi, key=lambda s: (len(s), s)):
        if not by_shape_lerobot.get(s):
            for k in by_shape_openpi[s]:
                _emit(report, f"  {k}  shape={s}")
    _emit(report)

    # Decision gate hint.
    _emit(report, "=== Decision gate ===")
    if len(matched) < 800:
        _emit(report, "  matched < 800 → STOP and analyze before Step C.")
    elif len(matched) >= 810 and (len(lerobot_only) + len(openpi_only)) <= 20:
        _emit(report, "  matched >= 810, orphans <= 20 → OK to proceed to Step C.")
        _emit(report, "  Fill EXTRA_RENAMES in convert_lerobot_to_openpi.py from the")
        _emit(report, "  'unambiguous rename' lines above.")
    else:
        _emit(report, "  in-between zone — review orphan list before proceeding.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report) + "\n")
    print(f"\n[audit] wrote report to {REPORT_PATH}")

    # Also dump the unambiguous rename suggestions as JSON for Step C to ingest.
    json_path = REPORT_PATH.with_suffix(".renames.json")
    suggestions: dict[str, str] = {}
    for s in all_shapes:
        lro = by_shape_lerobot.get(s, [])
        oro = by_shape_openpi.get(s, [])
        if len(lro) == 1 and len(oro) == 1:
            suggestions[lro[0]] = oro[0]
    json_path.write_text(json.dumps(suggestions, indent=2, sort_keys=True) + "\n")
    print(f"[audit] wrote unambiguous rename suggestions to {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
