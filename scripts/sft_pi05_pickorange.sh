#!/usr/bin/env bash
# SFT Pi 0.5 on the LightwheelAI/leisaac-pick-orange dataset (60 eps,
# ~36k frames, 30fps, front+wrist cams, so101_follower).
#
# Fresh run:
#   STEPS=30000 LOG_FREQ=50 bash scripts/sft_pi05_pickorange.sh
#
# Resume:
#   RESUME=true bash scripts/sft_pi05_pickorange.sh
#
# Recipe matches the prior sponge run:
#   - lerobot/pi05_base init (gemma_2b + 300M action expert)
#   - train_expert_only=true + freeze_vision_encoder=true → ~24GB
#   - bf16 + gradient_checkpointing
#   - empty_cameras=1 (pi05_base expects 3 cam slots; dataset gives 2)
#
# Camera rename: dataset → pi05_base canonical keys
#   observation.images.front → observation.images.base_0_rgb
#   observation.images.wrist → observation.images.right_wrist_0_rgb

set -euo pipefail

LEROBOT_BIN="/home/hlei/miniconda3/envs/rlinf-lerobot-train/bin/lerobot-train"
DATASET_REPO="LightwheelAI/leisaac-pick-orange"
# Local v3.0 conversion of the (originally v2.1) HF dataset. Produced by:
#   python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
#       --repo-id=LightwheelAI/leisaac-pick-orange \
#       --root=/home/hlei/.cache/huggingface/lerobot/LightwheelAI/leisaac-pick-orange \
#       --push-to-hub=False
DATASET_ROOT="${DATASET_ROOT:-/home/hlei/.cache/huggingface/lerobot/LightwheelAI/leisaac-pick-orange/LightwheelAI/leisaac-pick-orange}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sft_pi05_pickorange}"
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
  --dataset.root="$DATASET_ROOT" \
  --dataset.revision=main \
  --rename_map='{"observation.images.front": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.right_wrist_0_rgb"}' \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --num_workers=4 \
  --save_freq=2000 \
  --log_freq="$LOG_FREQ" \
  --output_dir="$OUTPUT_DIR" \
  --job_name=sft_pi05_pickorange \
  --policy.push_to_hub=false \
  --policy.repo_id=local/sft_pi05_pickorange \
  --wandb.enable=false
