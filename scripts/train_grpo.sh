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

ARM="${ARM:-failure_aware}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MODEL_PATH="${MODEL_PATH:-}"
VERL_ROOT="${VERL_ROOT:-$ROOT/third_party/verl}"
TRAIN_FILE="${TRAIN_FILE:-$ROOT/results/data/processed/rl/rl_${ARM}.parquet}"
VAL_FILE="${VAL_FILE:-$ROOT/results/data/processed/rl_dev/rl_uniform.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/tmp/checkpoints/grpo_${ARM}}"
STEPS="${STEPS:-$GRPO_STEPS}"
SKIP_SMOKE_VERIFICATION="${SKIP_SMOKE_VERIFICATION:-0}"
EXPORT_ONLY="${EXPORT_ONLY:-0}"
case "${GRPO_PARAM_OFFLOAD:-true}" in
  1|true|yes|on) PARAM_OFFLOAD=true ;;
  0|false|no|off) PARAM_OFFLOAD=false ;;
  *) echo "GRPO_PARAM_OFFLOAD must be a boolean" >&2; exit 1 ;;
esac
case "${GRPO_OPTIMIZER_OFFLOAD:-true}" in
  1|true|yes|on) OPTIMIZER_OFFLOAD=true ;;
  0|false|no|off) OPTIMIZER_OFFLOAD=false ;;
  *) echo "GRPO_OPTIMIZER_OFFLOAD must be a boolean" >&2; exit 1 ;;
esac
RUN_PHASE="${RUN_PHASE:-formal}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}"
METADATA_PATH="${RUN_METADATA_PATH:-$ROOT/tmp/artifacts/runs/${RUN_PHASE}_${ARM}_${RUN_ID}.json}"
PROFILE_PYTHON="$VERL_ROOT/.venv/bin/python"

if [[ "$EXPORT_ONLY" == "1" ]]; then
  latest_checkpoint="$OUTPUT_DIR/latest_checkpointed_iteration.txt"
  if [[ ! -f "$latest_checkpoint" || "$(tr -d '[:space:]' < "$latest_checkpoint")" != "$STEPS" ]]; then
    echo "EXPORT_ONLY requires a complete checkpoint at requested STEPS=$STEPS" >&2
    exit 1
  fi
  CHECKPOINT_SAVE_CONTENTS='["model","hf_model"]'
  export VERL_EXPORT_CHECKPOINT_AT_START=1
else
  CHECKPOINT_SAVE_CONTENTS='["model","optimizer","extra","hf_model"]'
  export VERL_EXPORT_CHECKPOINT_AT_START=0
fi

if [[ -z "$MODEL_PATH" ]]; then
  echo "MODEL_PATH must point to the exported SFT model" >&2
  exit 1
fi
if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "Missing RL prompt data: $TRAIN_FILE" >&2
  exit 1
fi
if [[ "$VAL_FILE" != "null" && ! -f "$VAL_FILE" ]]; then
  echo "Missing RL validation data: $VAL_FILE" >&2
  exit 1
fi
if [[ ! -x "$PROFILE_PYTHON" ]]; then
  echo "Missing veRL environment: $VERL_ROOT/.venv" >&2
  exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Missing GRPO model directory: $MODEL_PATH" >&2
  exit 1
fi

export DEEPSEEK_API_BASE="$DEEPSEEK_BASE_URL"
export DEEPSEEK_TELEMETRY_PATH="${DEEPSEEK_TELEMETRY_PATH:-$ROOT/tmp/artifacts/logs/deepseek_grpo_${ARM}.jsonl}"
mkdir -p "$(dirname "$DEEPSEEK_TELEMETRY_PATH")"

"$PROFILE_PYTHON" "$ROOT/scripts/run_metadata.py" validate >/dev/null
echo "GRPO_RUNTIME max_inflight_gen_batches=$GRPO_MAX_INFLIGHT_GEN_BATCHES agent_workers=$AGENT_WORKERS max_active_rollouts_per_worker=$TAU2_MAX_ACTIVE_ROLLOUTS_PER_WORKER"
"$PROFILE_PYTHON" - <<'PY'
from telecom_post_training.deepseek_client import DeepSeekClientConfig

config = DeepSeekClientConfig.from_env()
print(
    "DEEPSEEK_CLIENT_OK "
    f"process_concurrency={config.min_concurrency}-{config.max_concurrency} "
    f"global_cap={config.global_max_concurrency} "
    f"workers={config.worker_processes} "
    "gate_scope=process "
    f"fallback={'enabled' if config.fallback_enabled else 'disabled'} "
    f"fallback_global_cap={config.fallback_global_max_concurrency} "
    f"adaptive={int(config.adaptive_concurrency)} "
    f"timeout={config.request_timeout_seconds}s "
    f"max_retries={config.max_retries} "
    f"retry_jitter_ratio={config.retry_jitter_ratio} "
    f"retry_max_seconds={config.retry_max_seconds} "
    f"total_deadline={config.total_request_deadline_seconds}s"
)
PY

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
      --phase grpo-smoke \
      --marker "$ROOT/tmp/artifacts/logs/grpo_smoke.ok" \
      --require-clean
  fi
fi

"$PROFILE_PYTHON" "$ROOT/scripts/run_metadata.py" snapshot \
  --root "$ROOT" \
  --phase "$RUN_PHASE" \
  --arm "$ARM" \
  --data-file "$TRAIN_FILE" \
  --data-file "$ROOT/results/data/processed/rl/rl_uniform.parquet" \
  --data-file "$ROOT/results/data/processed/rl/rl_failure_aware.parquet" \
  --data-file "$ROOT/results/data/processed/rl_dev/rl_uniform.parquet" \
  --data-file "$ROOT/results/data/processed/rl_dev/rl_failure_aware.parquet" \
  --model-path "$MODEL_PATH" \
  --train-file "$TRAIN_FILE" \
  --val-file "$VAL_FILE" \
  --steps "$STEPS" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --output "$METADATA_PATH" >/dev/null

echo "GRPO_METADATA=$METADATA_PATH"
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
export TAU2_REPO="$ROOT/third_party/tau2-bench"
export TAU2_ROLLOUT_MAX_TURNS=30
export TAU2_MAX_TOKENS_PER_TURN=256
export TAU2_ENV_TIMEOUT=180
export DEEPSEEK_USER_TEMPERATURE=0.7
export DEEPSEEK_USER_SEED=42

cd "$VERL_ROOT"
"$VERL_ROOT/.venv/bin/python" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  algorithm.filter_groups.enable=true \
  algorithm.filter_groups.metric=reward \
  algorithm.filter_groups.max_inflight_gen_batches="$GRPO_MAX_INFLIGHT_GEN_BATCHES" \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.prompt_key=prompt \
  data.return_raw_chat=true \
  data.train_batch_size="$GRPO_TRAIN_BATCH_SIZE" \
  data.seed=20261103 \
  data.max_prompt_length=16384 \
  data.max_response_length=16384 \
  data.filter_overlong_prompts=true \
  data.truncation=error \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=64 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$GRPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$GRPO_MICRO_BATCH_SIZE" \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.fsdp_config.seed=20261103 \
  actor_rollout_ref.actor.fsdp_config.param_offload="$PARAM_OFFLOAD" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="$OPTIMIZER_OFFLOAD" \
  actor_rollout_ref.actor.checkpoint.save_contents="$CHECKPOINT_SAVE_CONTENTS" \
  actor_rollout_ref.ref.strategy=fsdp2 \
  actor_rollout_ref.ref.fsdp_config.seed=20261103 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$GRPO_LOGPROB_MICRO_BATCH_SIZE" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.seed=20261103 \
  actor_rollout_ref.rollout.prompt_length=16384 \
  actor_rollout_ref.rollout.response_length=16384 \
  actor_rollout_ref.rollout.n="$GRPO_GROUP_SIZE" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GRPO_VLLM_GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.max_num_seqs="$GRPO_VLLM_MAX_NUM_SEQS" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$GRPO_LOGPROB_MICRO_BATCH_SIZE" \
  actor_rollout_ref.rollout.calculate_log_probs=true \
  actor_rollout_ref.rollout.multi_turn.enable=true \
  actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=30 \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=30 \
  actor_rollout_ref.rollout.agent.default_agent_loop=tau2_telecom \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$ROOT/configs/verl_agent_loop.yaml" \
  actor_rollout_ref.rollout.agent.num_workers="$AGENT_WORKERS" \
  +data.apply_chat_template_kwargs.enable_thinking=false \
  trainer.project_name=telecom-agent-post-training \
  trainer.experiment_name="grpo-${ARM}" \
  +trainer.seed=20261103 \
  trainer.n_gpus_per_node="$NPROC_PER_NODE" \
  trainer.nnodes=1 \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.v1.sampler.sync_refill_failed_groups=true \
  trainer.total_training_steps="$STEPS" \
  trainer.save_freq=25 \
  trainer.test_freq=25 \
  trainer.val_before_train=false \
  trainer.logger='["console"]'
