#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARM="${ARM:?Set ARM to uniform or failure_aware}"
STEPS="${STEPS:-${GRPO_STEPS:-75}}"

case "$ARM" in
  uniform)
    MODEL_PATH="$ROOT/tmp/checkpoints/sft_uniform/global_step_25/huggingface"
    ;;
  failure_aware)
    MODEL_PATH="$ROOT/tmp/checkpoints/sft_failure_aware/global_step_56/huggingface"
    ;;
  *)
    echo "ARM must be uniform or failure_aware" >&2
    exit 1
    ;;
esac

echo "GRPO_EXPORT_ONLY arm=$ARM steps=$STEPS model=$MODEL_PATH"
EXPORT_ONLY=1 \
RUN_PHASE=export \
SKIP_SMOKE_VERIFICATION=1 \
ARM="$ARM" \
STEPS="$STEPS" \
MODEL_PATH="$MODEL_PATH" \
bash "$ROOT/scripts/train_grpo.sh"
