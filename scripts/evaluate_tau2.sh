#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_LABEL="${MODEL_LABEL:?Set MODEL_LABEL to base, uniform, or failure_aware}"
MODEL_ALIAS="${MODEL_ALIAS:-student}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
TASK_SPLIT="${TASK_SPLIT:-test}"
TASK_STEPS="${TASK_STEPS:-100}"
CONCURRENCY="${CONCURRENCY:-8}"

case "$TASK_SPLIT" in
  test)
    TAU2_SPLIT="test"
    EXPECTED_TASKS=40
    DEFAULT_SEEDS="300 301 302"
    ;;
  dev)
    TAU2_SPLIT="train"
    EXPECTED_TASKS=10
    DEFAULT_SEEDS="200 201 202"
    ;;
  *)
    echo "TASK_SPLIT must be dev or test" >&2
    exit 1
    ;;
esac

OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/tmp/artifacts/eval/$MODEL_LABEL/$TASK_SPLIT}"
NORMALIZED_ROOT="${NORMALIZED_ROOT:-$ROOT/tmp/data/eval}"
EVAL_PYTHON="${EVAL_PYTHON:-$ROOT/third_party/verl/.venv/bin/python}"
if [[ ! -x "$EVAL_PYTHON" ]]; then
  EVAL_PYTHON="$ROOT/.venv/bin/python"
fi

if [[ ! -f "$ROOT/.env" ]]; then
  echo "Missing $ROOT/.env" >&2
  exit 1
fi
if [[ ! -x "$EVAL_PYTHON" ]]; then
  echo "Missing tau2 environment: $EVAL_PYTHON" >&2
  exit 1
fi

set -a
source "$ROOT/.env"
set +a

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="$VLLM_BASE_URL"
export DEEPSEEK_API_BASE="$DEEPSEEK_BASE_URL"

"$EVAL_PYTHON" - <<'PY'
import os

from telecom_post_training.settings import validate_deepseek_base_url

validate_deepseek_base_url(os.environ.get("DEEPSEEK_BASE_URL", ""))
PY

mapfile -t TASK_IDS < <(
  "$EVAL_PYTHON" - <<PY
import json
from pathlib import Path

manifest = json.loads(
    (Path("$ROOT") / "data/manifests/telecom_manifest.json").read_text()
)
print("\n".join(manifest["splits"]["$TASK_SPLIT"]))
PY
)

if [[ "${#TASK_IDS[@]}" -ne "$EXPECTED_TASKS" ]]; then
  echo "Expected $EXPECTED_TASKS $TASK_SPLIT tasks, found ${#TASK_IDS[@]}" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT" "$NORMALIZED_ROOT"

read -r -a EVAL_SEEDS <<< "${EVAL_SEEDS:-$DEFAULT_SEEDS}"
normalized_files=()
for seed in "${EVAL_SEEDS[@]}"; do
  run_dir="$OUTPUT_ROOT/seed_$seed"
  user_llm_args="$(
    printf \
      '{"temperature":%s,"max_tokens":256,"seed":%s,"timeout":90,"num_retries":3,"extra_body":{"thinking":{"type":"disabled"}}}' \
      "${DEEPSEEK_USER_TEMPERATURE:-0.7}" "$seed"
  )"
  "$EVAL_PYTHON" -m telecom_post_training.tau2_cli run \
    --domain telecom \
    --task-set-name telecom \
    --task-split-name "$TAU2_SPLIT" \
    --task-ids "${TASK_IDS[@]}" \
    --agent llm_agent \
    --agent-llm "openai/$MODEL_ALIAS" \
    --agent-llm-args '{"temperature":0.0,"max_tokens":256,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
    --user user_simulator \
    --user-llm "deepseek/$DEEPSEEK_MODEL" \
    --user-llm-args "$user_llm_args" \
    --num-trials 1 \
    --max-steps "$TASK_STEPS" \
    --max-concurrency "$CONCURRENCY" \
    --seed "$seed" \
    --save-to "$run_dir" \
    --auto-resume \
    --log-level INFO

  "$EVAL_PYTHON" -m telecom_post_training.evaluation \
    --input "$run_dir/results.json" \
    --split "$TASK_SPLIT"

  "$EVAL_PYTHON" -m telecom_post_training.cli normalize \
    --input "$run_dir" \
    --output "$NORMALIZED_ROOT/${MODEL_LABEL}_${TASK_SPLIT}_seed_${seed}.jsonl" \
    --split "$TASK_SPLIT"
  normalized_files+=(
    "$NORMALIZED_ROOT/${MODEL_LABEL}_${TASK_SPLIT}_seed_${seed}.jsonl"
  )
done

all_episodes="$NORMALIZED_ROOT/${MODEL_LABEL}_${TASK_SPLIT}_all.jsonl"
cat "${normalized_files[@]}" > "$all_episodes"
"$EVAL_PYTHON" -m telecom_post_training.cli summarize \
  --input "$all_episodes" \
  --output "$ROOT/tmp/artifacts/reports/eval_${MODEL_LABEL}_${TASK_SPLIT}.md"
