#!/usr/bin/env bash
# Smoke test the SFT pipeline: same config as sft_pi05_sponge.sh but only
# 2 steps + batch_size=1 to verify model loads, dataset reads, optimizer
# steps, and a checkpoint can be written. Use before committing to the
# multi-hour main run.

OUTPUT_DIR="${OUTPUT_DIR:-outputs/sft_smoke}" \
STEPS=2 \
BATCH=1 \
exec "$(dirname "$0")/sft_pi05_sponge.sh"
