#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARM="${ARM:-uniform}"
LOG_DIR="${LOG_DIR:-$ROOT/tmp/artifacts/logs}"
LOG_FILE="$LOG_DIR/sft_smoke_${ARM}.log"
METADATA="$LOG_DIR/sft_smoke_metadata.json"
MARKER="$LOG_DIR/sft_smoke.ok"

mkdir -p "$LOG_DIR"
rm -f "$METADATA" "$MARKER"
STEPS=1 \
VAL_FILE=null \
RUN_PHASE=sft-smoke \
RUN_METADATA_PATH="$METADATA" \
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/tmp/checkpoints/smoke_sft_${ARM}}" \
bash "$ROOT/scripts/train_sft.sh" 2>&1 | tee "$LOG_FILE"

"$ROOT/third_party/verl/.venv/bin/python" \
  "$ROOT/scripts/run_metadata.py" mark-smoke \
  --snapshot "$METADATA" \
  --output "$MARKER"
echo "SFT_SMOKE_OK arm=$ARM steps=1 log=$LOG_FILE"
