#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARM="${ARM:-uniform}"
LOG_DIR="${LOG_DIR:-$ROOT/tmp/artifacts/logs}"
CONTEXT_AUDIT="$LOG_DIR/grpo_smoke_context.jsonl"
DEEPSEEK_AUDIT="$LOG_DIR/grpo_smoke_deepseek.jsonl"
TRAIN_LOG="$LOG_DIR/grpo_smoke_${ARM}.log"
AUDIT_OUTPUT="$LOG_DIR/grpo_smoke_audit.json"
METADATA="$LOG_DIR/grpo_smoke_metadata.json"
MARKER="$LOG_DIR/grpo_smoke.ok"
VAL_FILE="${VAL_FILE:-$ROOT/results/data/processed/rl_dev/rl_uniform.parquet}"
SMOKE_RUN_ID="${SMOKE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/tmp/checkpoints/smoke_grpo_${ARM}_${SMOKE_RUN_ID}}"

mkdir -p "$LOG_DIR"
rm -f "$CONTEXT_AUDIT" "$DEEPSEEK_AUDIT" "$AUDIT_OUTPUT" "$METADATA" "$MARKER"

STEPS=1 \
VAL_FILE="$VAL_FILE" \
RUN_PHASE=grpo-smoke \
OUTPUT_DIR="$OUTPUT_DIR" \
RUN_METADATA_PATH="$METADATA" \
TAU2_CONTEXT_AUDIT_PATH="$CONTEXT_AUDIT" \
DEEPSEEK_TELEMETRY_PATH="$DEEPSEEK_AUDIT" \
bash "$ROOT/scripts/train_grpo.sh" 2>&1 | tee "$TRAIN_LOG"

"$ROOT/third_party/verl/.venv/bin/python" \
  "$ROOT/scripts/audit_grpo_smoke.py" \
  --context-log "$CONTEXT_AUDIT" \
  --deepseek-log "$DEEPSEEK_AUDIT" \
  --train-log "$TRAIN_LOG" \
  --output "$AUDIT_OUTPUT"

"$ROOT/third_party/verl/.venv/bin/python" \
  "$ROOT/scripts/run_metadata.py" mark-smoke \
  --snapshot "$METADATA" \
  --output "$MARKER"
echo "GRPO_SMOKE_OK arm=$ARM steps=1 audit=$AUDIT_OUTPUT"
