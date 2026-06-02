#!/usr/bin/env bash
# Run sft_pi05_sponge.sh with a tensorboard server attached.
#
# Pipes lerobot-train stdout through _sft_tb_tail.py (regex-parses
# `key:value` metric tokens and writes scalars to a SummaryWriter), and
# starts a `tensorboard` server in the background pointing at the same
# logdir. The pipe also passes stdout through, so you still see the live
# lerobot output in the terminal.
#
# Open http://<host>:${TB_PORT:-6006} to monitor.
#
# Override anything via env: OUTPUT_DIR, STEPS, BATCH, LOG_FREQ, RESUME,
# TB_PORT, KEEP_CKPTS (default 2).

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sft_pi05_sponge}"
# tb artifacts live in a sibling dir — lerobot refuses to start if its
# output_dir already exists (and resume=False), so don't touch it.
TB_DIR="${OUTPUT_DIR}_tb"
TB_LOGDIR="$TB_DIR/events"
TB_PORT="${TB_PORT:-6006}"
KEEP_CKPTS="${KEEP_CKPTS:-3}"
CKPT_DIR="$OUTPUT_DIR/checkpoints"
PY="/home/hlei/miniconda3/envs/rlinf-lerobot-train/bin/python"
TB_BIN="/home/hlei/miniconda3/envs/rlinf-lerobot-train/bin/tensorboard"

mkdir -p "$TB_LOGDIR"

HOSTNAME_SHORT="$(hostname -s)"
HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"

echo "[run-with-tb] tb events: $TB_LOGDIR"
echo "[run-with-tb] starting tensorboard on :$TB_PORT (server log -> $TB_DIR/tb_server.log)"
"$TB_BIN" --logdir "$TB_LOGDIR" --port "$TB_PORT" --bind_all \
  > "$TB_DIR/tb_server.log" 2>&1 &
TB_PID=$!

# Checkpoint pruner: keep the newest $KEEP_CKPTS numeric ckpt dirs (the
# `last` symlink is preserved). Polls every 60s — cheap, and ckpts only
# appear every save_freq*updt_s seconds anyway.
(
  while true; do
    sleep 60
    [ -d "$CKPT_DIR" ] || continue
    mapfile -t old < <(ls -1 "$CKPT_DIR" 2>/dev/null \
      | grep -E '^[0-9]+$' | sort -n | head -n -"$KEEP_CKPTS")
    for d in "${old[@]:-}"; do
      [ -n "$d" ] && rm -rf "$CKPT_DIR/$d" \
        && echo "[prune] removed $CKPT_DIR/$d (keep last $KEEP_CKPTS)"
    done
  done
) &
PRUNE_PID=$!

trap "echo '[run-with-tb] stopping tb=$TB_PID prune=$PRUNE_PID'; \
      kill $TB_PID $PRUNE_PID 2>/dev/null || true" EXIT

echo "[run-with-tb] open one of:"
echo "                http://localhost:${TB_PORT}"
[ -n "$HOSTNAME_SHORT" ] && echo "                http://${HOSTNAME_SHORT}:${TB_PORT}"
[ -n "$HOST_IP" ]        && echo "                http://${HOST_IP}:${TB_PORT}"

# Forward overrides to the inner script. Defaults live there — only export
# what the user actually set, so `set -u` doesn't trip on unset vars here.
export OUTPUT_DIR
[ -n "${STEPS:-}" ]    && export STEPS    || true
[ -n "${BATCH:-}" ]    && export BATCH    || true
[ -n "${LOG_FREQ:-}" ] && export LOG_FREQ || true
[ -n "${RESUME:-}" ]   && export RESUME   || true
"$HERE/sft_pi05_sponge.sh" 2>&1 \
  | tee -a "$TB_DIR/run.log" \
  | "$PY" "$HERE/_sft_tb_tail.py" --logdir "$TB_LOGDIR"
