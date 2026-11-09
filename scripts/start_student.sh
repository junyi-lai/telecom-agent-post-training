#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-$ROOT/models/Qwen3.5-2B}"
PORT="${PORT:-8011}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.60}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "Missing model config: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -x "$ROOT/.venv/bin/vllm" ]]; then
  echo "Missing vLLM environment; run uv sync --extra collect" >&2
  exit 1
fi

export VLLM_USE_FLASHINFER_SAMPLER=0
exec "$ROOT/.venv/bin/vllm" serve "$MODEL_PATH" \
  --served-model-name student \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --enforce-eager \
  --safetensors-load-strategy prefetch \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --trust-remote-code
