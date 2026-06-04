"""Phase 1.5c — convert lerobot SFT ckpt to RLinf openpi-loadable format.

Writes:
  outputs/sft_pi05_sponge/openpi_remapped/
    model.safetensors                                       (remapped weights)
    aswinkumar99/.../norm_stats.json                        (real stats, copied from Step B output)

Remap rules (verified by scripts/audit_ckpt_keys.py output):
  1. Strip `model.` prefix from every key (covers 811/812).
  2. EXTRA_RENAMES — one entry: lm_head.weight → embed_tokens.weight
     (HF tie_word_embeddings; lerobot saves lm_head only, openpi
     parameter list exposes embed_tokens only, same (257152, 2048) tensor).

Buffers `rotary_emb.inv_freq` × 2 and `vision_model.embeddings.position_ids`
are deterministic and recomputed by HF transformers at init; we don't ship them.

Run with the RLinf .venv python:
    /home/hlei/RLinf/.venv/bin/python scripts/convert_lerobot_to_openpi.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import safetensors.torch as st

SRC_CKPT = Path(os.environ.get(
    "LEROBOT_CKPT",
    "/home/hlei/robotic/lerobot-rlinf/outputs/sft_pi05_sponge/checkpoints/010000/pretrained_model/model.safetensors",
))
OUT_DIR = Path(os.environ.get(
    "REMAP_OUT_DIR",
    "/home/hlei/robotic/lerobot-rlinf/outputs/sft_pi05_sponge/openpi_remapped",
))
ASSET_SUBPATH = os.environ.get(
    "ASSET_SUBPATH",
    "aswinkumar99/LeRobot-SO101-task1-single-sponge-no-distractors-random-locations",
)
NORM_STATS_SRC = Path(os.environ.get(
    "NORM_STATS_SRC",
    str(Path("/home/hlei/RLinf/assets/pi05_isaaclab_so101_lift") / ASSET_SUBPATH / "norm_stats.json"),
))
AUDIT_RENAMES_JSON = Path(os.environ.get(
    "AUDIT_RENAMES_JSON",
    "/home/hlei/robotic/lerobot-rlinf/outputs/sft_pi05_sponge/key_audit.renames.json",
))

# Filled from key_audit.renames.json (the shape-unambiguous orphan pairs).
# This is the only manual rename beyond the bulk `model.` prefix strip.
EXTRA_RENAMES: dict[str, str] = {
    "model.paligemma_with_expert.paligemma.lm_head.weight":
        "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight",
}

# Buffers the openpi model materializes itself (rotary inv_freq, position_ids).
# Listed here so the dry-run validation knows it's OK that they're missing.
OPENPI_DETERMINISTIC_BUFFERS = {
    "paligemma_with_expert.paligemma.model.language_model.rotary_emb.inv_freq",
    "paligemma_with_expert.gemma_expert.model.rotary_emb.inv_freq",
    "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.position_ids",
}


def remap(src: dict) -> dict:
    out: dict = {}
    for k, v in src.items():
        if k in EXTRA_RENAMES:
            nk = EXTRA_RENAMES[k]
        else:
            nk = k.removeprefix("model.")
        if nk in out:
            raise RuntimeError(f"rename collision: two source keys map to {nk}")
        out[nk] = v
    return out


def main() -> int:
    # Sanity: confirm EXTRA_RENAMES matches the audit suggestions on disk.
    if AUDIT_RENAMES_JSON.exists():
        audit_suggestions = json.loads(AUDIT_RENAMES_JSON.read_text())
        if audit_suggestions != EXTRA_RENAMES:
            print(
                "[convert] WARNING: EXTRA_RENAMES in this script differs from the\n"
                "          unambiguous rename suggestions in key_audit.renames.json.\n"
                f"          script: {EXTRA_RENAMES}\n"
                f"          audit:  {audit_suggestions}\n"
                "          Re-run audit and reconcile if intentional."
            )
    else:
        print("[convert] note: audit renames json not found, skipping reconciliation check")

    print(f"[convert] loading {SRC_CKPT}")
    src = st.load_file(str(SRC_CKPT), device="cpu")
    print(f"[convert] loaded {len(src)} tensors")

    remapped = remap(src)
    print(f"[convert] remapped to {len(remapped)} tensors")

    # Dry-run validation: compare against openpi expected keys (from audit module).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from audit_ckpt_keys import load_openpi_shapes  # type: ignore
    print("[convert] building meta-device openpi model for key validation")
    expected = load_openpi_shapes()

    remapped_keys = set(remapped)
    expected_keys = set(expected)

    missing = sorted(expected_keys - remapped_keys)
    unexpected = sorted(remapped_keys - expected_keys)
    shape_bad: list[tuple[str, tuple, tuple]] = []
    for k in remapped_keys & expected_keys:
        if tuple(remapped[k].shape) != expected[k]:
            shape_bad.append((k, tuple(remapped[k].shape), expected[k]))

    print(f"[convert] missing in remap (vs openpi expected): {len(missing)}")
    for k in missing:
        marker = "  (deterministic buffer, OK)" if k in OPENPI_DETERMINISTIC_BUFFERS else "  !!"
        print(f"    {k}{marker}")
    print(f"[convert] unexpected in remap (no openpi target): {len(unexpected)}")
    for k in unexpected:
        print(f"    {k}  !!")
    if shape_bad:
        print(f"[convert] !!! SHAPE MISMATCH on {len(shape_bad)} keys:")
        for k, rs, es in shape_bad:
            print(f"    {k}: remap {rs} vs openpi {es}")

    surprising_missing = [k for k in missing if k not in OPENPI_DETERMINISTIC_BUFFERS]
    assert not unexpected, f"unexpected keys in remap: {unexpected}"
    assert not shape_bad, f"shape mismatches: {shape_bad}"
    assert not surprising_missing, f"unexpected missing keys: {surprising_missing}"
    print("[convert] validation OK (all missing are deterministic buffers)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_safetensors = OUT_DIR / "model.safetensors"
    print(f"[convert] writing {out_safetensors}")
    st.save_file(remapped, str(out_safetensors))

    out_norm = OUT_DIR / ASSET_SUBPATH / "norm_stats.json"
    out_norm.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(NORM_STATS_SRC, out_norm)
    print(f"[convert] copied norm_stats -> {out_norm}")

    sz = out_safetensors.stat().st_size / (1024**3)
    print(f"\n[convert] done. ckpt size: {sz:.2f} GB")
    print(f"[convert] point RLinf cfg.model_path at: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
