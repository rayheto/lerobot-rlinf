"""Phase 1.5b — extract real norm_stats.json from lerobot SFT preprocessor.

Reads policy_preprocessor_step_2_normalizer_processor.safetensors (flat
format: `<feature>.<stat>` keys with shape (6,) for SO-101) and writes
openpi-format norm_stats.json (nested: {"norm_stats": {"state": {...},
"actions": {...}}}).

Overwrites the two dummy norm_stats.json locations under the RLinf assets
dir and the HF cache. Backs up the original dummy as .dummy.bak on first run.

Run with the RLinf .venv python:
    /home/hlei/RLinf/.venv/bin/python scripts/extract_norm_stats.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import safetensors.torch as st

LEROBOT_NORM = Path(os.environ.get(
    "LEROBOT_NORM",
    "/home/hlei/robotic/lerobot-rlinf/outputs/sft_pi05_sponge/checkpoints/010000/pretrained_model/policy_preprocessor_step_2_normalizer_processor.safetensors",
))

ASSET_SUBPATH = os.environ.get(
    "ASSET_SUBPATH",
    "aswinkumar99/LeRobot-SO101-task1-single-sponge-no-distractors-random-locations",
)

# Comma-separated extra target paths via env; defaults are the two known
# norm_stats locations RLinf openpi reads on actor init.
_default_targets = [
    str(Path("/home/hlei/RLinf/assets/pi05_isaaclab_so101_lift") / ASSET_SUBPATH / "norm_stats.json"),
    str(
        Path("/home/hlei/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/9e55186ad36e66b95cda57bc47818d9e6237ae30")
        / ASSET_SUBPATH
        / "norm_stats.json"
    ),
]
TARGETS = [Path(p) for p in os.environ.get("NORM_STATS_TARGETS", ",".join(_default_targets)).split(",") if p]

# (lerobot feature name, openpi norm_stats key)
FEATURES = [
    ("observation.state", "state"),
    ("action", "actions"),
]

STATS = ["mean", "std", "q01", "q99"]


def build_norm_stats() -> dict:
    sd = st.load_file(str(LEROBOT_NORM))
    out: dict[str, dict[str, list[float]]] = {}
    for lerobot_feat, openpi_key in FEATURES:
        entry: dict[str, list[float]] = {}
        for stat in STATS:
            key = f"{lerobot_feat}.{stat}"
            if key not in sd:
                raise KeyError(f"missing {key} in {LEROBOT_NORM}")
            t = sd[key]
            if t.ndim != 1:
                raise ValueError(f"expected 1-D tensor for {key}, got shape {tuple(t.shape)}")
            entry[stat] = [float(x) for x in t.tolist()]
        out[openpi_key] = entry
    return {"norm_stats": out}


def write_target(payload: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        backup = target.with_suffix(target.suffix + ".dummy.bak")
        if not backup.exists():
            shutil.copy2(target, backup)
            print(f"  backed up existing dummy -> {backup}")
        else:
            print(f"  backup already exists at {backup}, skipping")
    target.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"  wrote {target}")


def verify_with_openpi(target: Path) -> None:
    from openpi.shared.normalize import load as _load
    stats = _load(target.parent)
    if set(stats) != {"state", "actions"}:
        raise RuntimeError(f"openpi load returned keys {set(stats)}")
    print(
        f"  openpi.shared.normalize.load OK — "
        f"state.mean[:3]={stats['state'].mean[:3].tolist()}, "
        f"actions.q99[:3]={stats['actions'].q99[:3].tolist()}"
    )


def main() -> int:
    print(f"[extract] reading {LEROBOT_NORM}")
    payload = build_norm_stats()

    # Sanity print
    for k, v in payload["norm_stats"].items():
        print(f"  {k}: mean[:3]={v['mean'][:3]} std[:3]={v['std'][:3]} "
              f"q01[:3]={v['q01'][:3]} q99[:3]={v['q99'][:3]}  (dim={len(v['mean'])})")

    for target in TARGETS:
        print(f"\n[extract] target: {target}")
        write_target(payload, target)
        verify_with_openpi(target)

    print("\n[extract] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
