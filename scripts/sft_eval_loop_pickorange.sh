#!/usr/bin/env bash
# Orchestrator: 在每个 save_freq ckpt 出现后暂停训练 → lerobot-native eval
# → 恢复训练，循环到 40k。
#
# 用法：
#   nohup bash scripts/sft_eval_loop_pickorange.sh > /tmp/sft_eval_loop.log 2>&1 &
#
# 无需 remap：直接 PI05Policy.from_pretrained(ckpt/pretrained_model)
# 评估结果写入 outputs/sft_pi05_pickorange/eval_results.tsv

set -uo pipefail

REPO=/home/hlei/robotic/lerobot-rlinf
cd "$REPO"

CKPT_DIR=outputs/sft_pi05_pickorange/checkpoints
TARGET_STEPS=(30000 32000 34000 36000 38000 40000)
EVAL_TSV=outputs/sft_pi05_pickorange/eval_results.tsv
ISAAC_PY=/home/hlei/miniconda3/envs/rlinf-isaacsim-env/bin/python

mkdir -p "$(dirname "$EVAL_TSV")"
[ -f "$EVAL_TSV" ] || printf "step\tplaced\ttotal\tsr_pct\tlog\n" > "$EVAL_TSV"

# ---------- helpers ----------

TRAINER_PIDFILE=/tmp/sft_pi05_pickorange.trainer.pid

start_trainer() {
  local logfile=$1
  if [ -f "$TRAINER_PIDFILE" ]; then
    local existing
    existing=$(cat "$TRAINER_PIDFILE" 2>/dev/null || true)
    if [ -n "$existing" ] && kill -0 "$existing" 2>/dev/null; then
      if grep -aq "lerobot-train" "/proc/$existing/cmdline" 2>/dev/null; then
        echo "[orch] adopting existing trainer pid=$existing" >&2
        echo "$existing"
        return 0
      fi
    fi
    rm -f "$TRAINER_PIDFILE"
  fi
  echo "[orch] starting trainer, log=$logfile" >&2
  setsid nohup bash -c "
    exec /home/hlei/miniconda3/envs/rlinf-lerobot-train/bin/lerobot-train \\
      --config_path=$REPO/outputs/sft_pi05_pickorange/checkpoints/last/pretrained_model/train_config.json \\
      --resume=true
  " > "$logfile" 2>&1 &
  local pid=$!
  echo "$pid" > "$TRAINER_PIDFILE"
  echo "$pid"
}

wait_for_ckpt() {
  local step_padded=$1
  local pid=$2
  local target="$CKPT_DIR/$step_padded"
  while [ ! -d "$target" ]; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[orch] trainer pid=$pid died before $target appeared" >&2
      return 1
    fi
    sleep 30
  done
  sleep 20
  return 0
}

stop_trainer() {
  local pid=$1
  echo "[orch] stopping trainer pid=$pid (SIGINT)"
  kill -INT "$pid" 2>/dev/null || true
  for _ in $(seq 1 60); do
    if ! kill -0 "$pid" 2>/dev/null; then break; fi
    sleep 1
  done
  pkill -INT -f "lerobot-train --config_path" 2>/dev/null || true
  sleep 5
  pkill -9 -f "lerobot-train --config_path" 2>/dev/null || true
  sleep 5
  rm -f "$TRAINER_PIDFILE"
}

run_eval() {
  local step=$1
  local log="/tmp/eval_pickorange_${step}.log"
  local model_path="$CKPT_DIR/$(printf "%06d" "$step")/pretrained_model"
  echo "[orch] eval step=$step model=$model_path → $log"

  rm -rf /tmp/isaaclab/logs/dataset.hdf5* 2>/dev/null || true

  "$ISAAC_PY" scripts/eval_pi05_pickorange_lerobot.py \
      --num-envs 2 --episodes 20 --max-steps 1500 \
      --model-path "$model_path" \
      > "$log" 2>&1 || true

  local sr_line placed total sr_pct
  sr_line=$(grep -Po 'SR: \d+/\d+' "$log" || true)
  placed=$(echo "$sr_line" | grep -oP 'SR: \K\d+')
  total=$(echo "$sr_line" | grep -oP '/\K\d+')
  sr_pct=$(echo "$sr_line" | awk -F'[/ ]+' '{if ($3>0) printf "%.1f", $2/$3*100; else print "0"}')

  printf "%d\t%s\t%s\t%s\t%s\n" "$step" "${placed:-0}" "${total:-0}" "${sr_pct:-0}" "$log" >> "$EVAL_TSV"
  echo "[orch] step=$step SR=${placed:-0}/${total:-0} = ${sr_pct:-0}%"
}

# ---------- main loop ----------

for STEP in "${TARGET_STEPS[@]}"; do
  STEP_PADDED=$(printf "%06d" "$STEP")
  CKPT_PATH="$CKPT_DIR/$STEP_PADDED"

  if [ ! -d "$CKPT_PATH" ]; then
    TRAIN_LOG="/tmp/sft_orch_${STEP_PADDED}.log"
    PID=$(start_trainer "$TRAIN_LOG")
    if ! wait_for_ckpt "$STEP_PADDED" "$PID"; then
      echo "[orch] FATAL: trainer died, see $TRAIN_LOG" >&2
      exit 1
    fi
    stop_trainer "$PID"
  else
    echo "[orch] $CKPT_PATH already exists, skipping training"
  fi

  if awk -F'\t' -v s="$STEP" 'NR>1 && $1==s {found=1} END{exit !found}' "$EVAL_TSV"; then
    echo "[orch] step=$STEP already in $EVAL_TSV, skipping eval"
    continue
  fi

  run_eval "$STEP"
done

echo "[orch] all targets done."
