#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  source "$ROOT/.env"
  set +a
fi

# shellcheck source=scripts/lib_profile.sh
source "$ROOT/scripts/lib_profile.sh"
load_training_profile "$ROOT"

ARM="${ARM:-uniform}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MODEL_PATH="${MODEL_PATH:-$ROOT/models/Qwen3.5-2B}"
VERL_ROOT="${VERL_ROOT:-$ROOT/third_party/verl}"
TRAIN_FILE="${TRAIN_FILE:-$ROOT/results/data/processed/sft/sft_${ARM}.parquet}"
VAL_FILE="${VAL_FILE:-null}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/tmp/checkpoints/sft_${ARM}}"
STEPS="${STEPS:-}"
SKIP_SMOKE_VERIFICATION="${SKIP_SMOKE_VERIFICATION:-0}"
RUN_PHASE="${RUN_PHASE:-formal}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}"
METADATA_PATH="${RUN_METADATA_PATH:-$ROOT/tmp/artifacts/runs/${RUN_PHASE}_${ARM}_${RUN_ID}.json}"
PROFILE_PYTHON="$VERL_ROOT/.venv/bin/python"

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "Missing SFT data: $TRAIN_FILE" >&2
  exit 1
fi
if [[ ! -x "$VERL_ROOT/.venv/bin/torchrun" || ! -x "$PROFILE_PYTHON" ]]; then
  echo "Missing veRL environment: $VERL_ROOT/.venv" >&2
  exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Missing model directory: $MODEL_PATH" >&2
  exit 1
fi

"$PROFILE_PYTHON" "$ROOT/scripts/run_metadata.py" validate >/dev/null

if [[ "$STEPS" != "1" ]]; then
  if [[ "$SKIP_SMOKE_VERIFICATION" == "1" ]]; then
    echo "SMOKE_ACCEPTANCE=skipped_by_user"
    "$PROFILE_PYTHON" "$ROOT/scripts/preflight.py" \
      --root "$ROOT" \
      --skip-gpu \
      --skip-credentials \
      --skip-runtime \
      --output "$ROOT/tmp/artifacts/preflight_formal.json"
  else
    "$PROFILE_PYTHON" "$ROOT/scripts/run_metadata.py" verify-smoke \
      --root "$ROOT" \
      --phase sft-smoke \
      --marker "$ROOT/tmp/artifacts/logs/sft_smoke.ok" \
      --require-clean
  fi
fi

"$PROFILE_PYTHON" "$ROOT/scripts/run_metadata.py" snapshot \
  --root "$ROOT" \
  --phase "$RUN_PHASE" \
  --arm "$ARM" \
  --data-file "$TRAIN_FILE" \
  --data-file "$ROOT/results/data/processed/sft/sft_uniform.parquet" \
  --data-file "$ROOT/results/data/processed/sft/sft_failure_aware.parquet" \
  --model-path "$MODEL_PATH" \
  --train-file "$TRAIN_FILE" \
  --val-file "$VAL_FILE" \
  --steps "${STEPS:-null}" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --output "$METADATA_PATH" >/dev/null

echo "SFT_METADATA=$METADATA_PATH"
"$PROFILE_PYTHON" - "$METADATA_PATH" <<'PY'
import json
import sys

metadata = json.load(open(sys.argv[1], encoding="utf-8"))
print(
    "RUN_ACCEPTANCE "
    f"smoke_acceptance={metadata['smoke_acceptance']} "
    f"git_commit={metadata['git_commit']} "
    f"effective_config_hash={metadata['effective_config_hash']} "
    "data_sha256="
    + json.dumps(metadata["data_sha256"], sort_keys=True, separators=(",", ":"))
)
PY
cd "$VERL_ROOT"
"$VERL_ROOT/.venv/bin/torchrun" \
  --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
  -m verl.trainer.sft_trainer \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size="$SFT_TRAIN_BATCH_SIZE" \
  data.micro_batch_size_per_gpu="$SFT_MICRO_BATCH_SIZE" \
  data.max_token_len_per_gpu="$SFT_MAX_TOKEN_LEN_PER_GPU" \
  data.use_dynamic_bsz=true \
  data.messages_key=messages \
  data.tools_key=tools \
  data.enable_thinking_key=enable_thinking \
  data.enable_thinking_default=false \
  data.max_length="$SFT_MAX_LENGTH" \
  data.truncation=error \
  data.ignore_input_ids_mismatch=true \
  model.path="$MODEL_PATH" \
  model.use_remove_padding=true \
  model.enable_gradient_checkpointing=true \
  model.lora_rank=32 \
  model.lora_alpha=64 \
  model.target_modules=all-linear \
  optim.lr=1e-4 \
  engine=fsdp \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.project_name=telecom-agent-post-training \
  trainer.experiment_name="sft-${ARM}" \
  trainer.seed=20261103 \
  trainer.total_epochs=2 \
  trainer.total_training_steps="${STEPS:-null}" \
  trainer.save_freq=25 \
  trainer.test_freq=-1 \
  checkpoint.save_contents='["model","optimizer","extra","hf_model"]' \
  trainer.logger='["console"]'
