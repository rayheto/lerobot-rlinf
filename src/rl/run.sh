#!/usr/bin/env bash
# Thin launcher for src/rl/train.py — mirrors RLinf's run_embodiment.sh but
# strips OmniGibson / Behavior cruft we don't need and pins paths to this
# repo's submodule layout.
#
# Usage:
#   bash src/rl/run.sh <config_name> [extra hydra overrides...]
#
# Examples:
#   bash src/rl/run.sh pick_orange_ppo_dryrun
#   bash src/rl/run.sh pick_orange_ppo runner.max_steps=1000
#   bash src/rl/run.sh pick_orange_ppo_eval +foo.bar=baz
#
# Requires: conda env rlinf-isaacsim-env (Isaac Sim 5.1 + leisaac).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
CONFIG_NAME="${1:-pick_orange_ppo_dryrun}"
shift || true

export EMBODIED_PATH="${REPO_ROOT}/third_party/RLinf/examples/embodiment"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/RLinf:${REPO_ROOT}/third_party/openpi/src:${REPO_ROOT}/third_party/openpi/packages/openpi-client/src:${PYTHONPATH:-}"
# Headless GL backend — Isaac Sim cameras + any mujoco-style renderers.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
# leisaac's constant.py picks the OUTER git root as ASSETS_ROOT, which
# resolves wrong in this nested submodule layout. Pin it.
export LEISAAC_ASSETS_ROOT="${LEISAAC_ASSETS_ROOT:-${REPO_ROOT}/third_party/leisaac/assets}"
# Memory: opt into bnb 8-bit AdamW (sitecustomize swaps torch.optim.AdamW)
# and enable expandable_segments to reduce fragmentation under tight VRAM.
# DSRL doesn't need bnb (only ~500K trainable params, fp32 AdamW is fine).
# PPO configs that hit the 24GB ceiling can re-enable by setting it on the env.
export RLINF_BNB_ADAMW8BIT="${RLINF_BNB_ADAMW8BIT:-0}"
# Skip FSDP wrap on single-GPU + no_shard runs. FSDP provides no value
# there (no sharding, no all-gather) but its flat_param/use_orig_params
# state corrupts under our custom CPU↔GPU offload, breaking training at
# ~step 10. Bypass replaces wrap_model with model.to(device) and walks
# named_parameters() for offload. See src/rl/dsrl_fsdp_bypass.py.
export RLINF_FSDP_BYPASS="${RLINF_FSDP_BYPASS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Cap thread fan-out — BLAS/OMP/MKL each default to ncores, and Isaac Sim +
# Ray workers each multiply that. On a single GPU dryrun we don't benefit
# from CPU parallelism; capping prevents machine-wide lag.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

ISAAC_PY="${ISAAC_PY:-/home/hlei/miniconda3/envs/rlinf-isaacsim-env/bin/python}"
if [[ ! -x "${ISAAC_PY}" ]]; then
    echo "isaac venv python not found at ${ISAAC_PY}; set ISAAC_PY=..." >&2
    exit 2
fi

LOG_DIR="${REPO_ROOT}/logs/$(date +'%Y%m%d-%H%M%S')-${CONFIG_NAME}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run.log"

CMD=(
    "${ISAAC_PY}" -u -m src.rl.train
    "--config-name=${CONFIG_NAME}"
    "runner.logger.log_path=${LOG_DIR}"
    "$@"
)
printf '%s\n' "${CMD[*]}" | tee "${LOG_FILE}"
exec "${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
