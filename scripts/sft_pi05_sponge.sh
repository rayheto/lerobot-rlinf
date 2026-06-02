#!/usr/bin/env bash
# SFT Pi 0.5 on the aswinkumar99 sponge dataset.
#
# Fresh run:
#   STEPS=10000 LOG_FREQ=5 bash scripts/sft_pi05_sponge.sh
#
# Resume from the latest checkpoint in $OUTPUT_DIR (only --resume + --config_path
# are passed; all other settings come from the saved train_config.json):
#   RESUME=true bash scripts/sft_pi05_sponge.sh
#
# Recipe (fresh runs only):
#   - Init from lerobot/pi05_base (gemma_2b + gemma_300m action expert).
#   - Freeze VLM + vision encoder; train only the 300M action expert + projections
#     ("train_expert_only=true" + "freeze_vision_encoder=true"). Keeps SFT under 24GB.
#   - bf16 + gradient checkpointing for further memory headroom.
#   - revision=main on the dataset to bypass lerobot's missing v3.0 git-tag check.
#
# Outputs: outputs/sft_pi05_sponge/ (checkpoints, logs, last/pretrained_model).

set -euo pipefail

LEROBOT_BIN="/home/hlei/miniconda3/envs/rlinf-lerobot-train/bin/lerobot-train"
DATASET_REPO="aswinkumar99/LeRobot-SO101-task1-single-sponge-no-distractors-random-locations"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sft_pi05_sponge}"
STEPS="${STEPS:-30000}"
BATCH="${BATCH:-8}"
LOG_FREQ="${LOG_FREQ:-50}"
RESUME="${RESUME:-false}"

if [ "$RESUME" = "true" ]; then
  CFG="$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json"
  if [ ! -f "$CFG" ]; then
    echo "[resume] no checkpoint found at $CFG" >&2
    exit 1
  fi
  echo "[resume] continuing from $CFG"
  exec "$LEROBOT_BIN" --config_path="$CFG" --resume=true
fi

"$LEROBOT_BIN" \
  --policy.path=lerobot/pi05_base \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.device=cuda \
  --policy.empty_cameras=1 \
  --dataset.repo_id="$DATASET_REPO" \
  --dataset.revision=main \
  --rename_map='{"observation.images.overhead": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.right_wrist_0_rgb"}' \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --num_workers=4 \
  --save_freq=2000 \
  --log_freq="$LOG_FREQ" \
  --output_dir="$OUTPUT_DIR" \
  --job_name=sft_pi05_sponge \
  --policy.push_to_hub=false \
  --policy.repo_id=local/sft_pi05_sponge \
  --wandb.enable=false
