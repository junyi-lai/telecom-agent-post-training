from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_sft_script_has_reproducible_seed_and_epoch_checkpoints() -> None:
    script = (ROOT / "scripts" / "train_sft.sh").read_text(encoding="utf-8")
    assert "trainer.seed=20261103" in script
    assert "trainer.save_freq=25" in script
    assert "trainer.total_epochs=2" in script
    assert '"$SFT_TRAIN_BATCH_SIZE"' in script
    assert '"$SFT_MICRO_BATCH_SIZE"' in script
    assert '"$SFT_MAX_TOKEN_LEN_PER_GPU"' in script
    assert '"$SFT_MAX_LENGTH"' in script
    assert "verify-smoke" in script
    assert "sft-smoke" in script


def test_grpo_script_seeds_trainer_data_rollout_and_fsdp() -> None:
    script = (ROOT / "scripts" / "train_grpo.sh").read_text(encoding="utf-8")
    assert 'STEPS="${STEPS:-$GRPO_STEPS}"' in script
    assert 'data.train_batch_size="$GRPO_TRAIN_BATCH_SIZE"' in script
    assert "data.seed=20261103" in script
    assert 'actor_rollout_ref.rollout.n="$GRPO_GROUP_SIZE"' in script
    assert "actor_rollout_ref.rollout.seed=20261103" in script
    assert "actor_rollout_ref.actor.fsdp_config.seed=20261103" in script
    assert "actor_rollout_ref.ref.fsdp_config.seed=20261103" in script
    assert "trainer.seed=20261103" in script
    assert "trainer.v1.sampler.sync_refill_failed_groups=true" in script
    assert (
        'algorithm.filter_groups.max_inflight_gen_batches='
        '"$GRPO_MAX_INFLIGHT_GEN_BATCHES"' in script
    )
    assert (
        'GRPO_RUNTIME max_inflight_gen_batches='
        "$GRPO_MAX_INFLIGHT_GEN_BATCHES" in script
    )
    assert "$TAU2_MAX_ACTIVE_ROLLOUTS_PER_WORKER" in script
    assert 'EXPORT_ONLY="${EXPORT_ONLY:-0}"' in script
    assert "VERL_EXPORT_CHECKPOINT_AT_START" in script
    assert 'checkpoint.save_contents="$CHECKPOINT_SAVE_CONTENTS"' in script
    assert 'trainer.total_training_steps="$STEPS"' in script
    assert "verify-smoke" in script
    assert "grpo-smoke" in script
    assert "+data.apply_chat_template_kwargs.enable_thinking=false" in script


def test_final_config_keeps_frozen_budget_and_task_cap() -> None:
    config = (ROOT / "configs" / "final_recommended.yaml").read_text(
        encoding="utf-8"
    )
    assert "training_pool_tasks: 164" in config
    assert "frozen_test_tasks: 40" in config
    assert "assistant_tokens_per_arm: 160000" in config
    assert "two_stage_coverage:" in config
    assert "save_freq_steps: 25" in config
    assert "max_examples_per_task: 2" in config
    assert "optimizer_steps: 75" in config
    assert "group_size: 8" in config
    assert "max_inflight_gen_batches: 1" in config
    assert "max_inflight_trajectories: 64" in config
    assert "max_active_rollouts_per_worker: 2" in config
    assert "max_simultaneous_rollouts: 16" in config
    assert "weighted_extra_rows: 164" in config
    assert "seed: 20261103" in config


def test_evaluation_script_keeps_student_greedy_and_varies_user_seed() -> None:
    script = (ROOT / "scripts" / "evaluate_tau2.sh").read_text(encoding="utf-8")
    assert 'TASK_SPLIT="${TASK_SPLIT:-test}"' in script
    assert 'TAU2_SPLIT="train"' in script
    assert "EXPECTED_TASKS=10" in script
    assert 'DEFAULT_SEEDS="200 201 202"' in script
    assert 'DEFAULT_SEEDS="300 301 302"' in script
    assert '"temperature":0.0' in script
    assert '"temperature":%s' in script
    assert '"seed":%s' in script
    assert '"${DEEPSEEK_USER_TEMPERATURE:-0.7}" "$seed"' in script
    assert "telecom_post_training.tau2_cli" in script


def test_grpo_script_configures_deepseek_gate_and_audit_paths() -> None:
    script = (ROOT / "scripts" / "train_grpo.sh").read_text(encoding="utf-8")
    assert "load_training_profile" in script
    assert "DeepSeekClientConfig" in script
    assert "DEEPSEEK_TELEMETRY_PATH" in script
    assert "DEEPSEEK_CLIENT_OK" in script
    assert "config.min_concurrency" in script
    assert "config.adaptive_concurrency" in script
    assert "config.retry_jitter_ratio" in script
    assert "config.global_max_concurrency" in script
    assert "config.fallback_enabled" in script
    assert 'param_offload="$PARAM_OFFLOAD"' in script
    assert 'optimizer_offload="$OPTIMIZER_OFFLOAD"' in script


def test_smoke_scripts_use_one_step_and_real_grpo_group() -> None:
    sft = (ROOT / "scripts" / "smoke_sft.sh").read_text(encoding="utf-8")
    grpo = (ROOT / "scripts" / "smoke_grpo.sh").read_text(encoding="utf-8")
    training = (ROOT / "scripts" / "train_grpo.sh").read_text(encoding="utf-8")
    assert "STEPS=1" in sft
    assert "STEPS=1" in grpo
    assert "rl_dev/rl_uniform.parquet" in grpo
    assert 'data.train_batch_size="$GRPO_TRAIN_BATCH_SIZE"' in training
    assert 'actor_rollout_ref.rollout.n="$GRPO_GROUP_SIZE"' in training
    assert "audit_grpo_smoke.py" in grpo
    assert "OUTPUT_DIR=\"${OUTPUT_DIR:-$ROOT/tmp/checkpoints/smoke_grpo_" in grpo


def test_grpo_export_uses_resume_only_and_hf_model_contents() -> None:
    exporter = (ROOT / "scripts" / "export_grpo_hf.sh").read_text(
        encoding="utf-8"
    )
    trainer = (ROOT / "scripts" / "train_grpo.sh").read_text(encoding="utf-8")
    assert "EXPORT_ONLY=1" in exporter
    assert "RUN_PHASE=export" in exporter
    assert "SKIP_SMOKE_VERIFICATION=1" in exporter
    assert "global_step_56/huggingface" in exporter
    assert (
        'CHECKPOINT_SAVE_CONTENTS=\'["model","hf_model"]\''
        in trainer
    )
    assert (
        'CHECKPOINT_SAVE_CONTENTS=\'["model","optimizer","extra","hf_model"]\''
        in trainer
    )


def test_evaluation_budget_matches_executed_protocol() -> None:
    import yaml

    config = yaml.safe_load((ROOT / "configs/evaluation.yaml").read_text())
    assert config["run"]["max_steps"] == 100
    assert config["dataset"]["seeds"] == [300, 301, 302]
    assert {model["name"] for model in config["models"]} == {
        "base", "uniform_sft", "failure_aware_sft",
        "uniform_grpo", "failure_aware_grpo",
    }


def test_student_server_uses_verified_16gb_settings() -> None:
    script = (ROOT / "scripts" / "start_student.sh").read_text(
        encoding="utf-8"
    )
    assert "VLLM_USE_FLASHINFER_SAMPLER=0" in script
    assert "--max-model-len 32768" in script
    assert '--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"' in script
    assert "--max-num-seqs \"$MAX_NUM_SEQS\"" in script
    assert "--enforce-eager" in script
    assert "--safetensors-load-strategy prefetch" in script
    assert "--enable-auto-tool-choice" in script
    assert "--tool-call-parser qwen3_coder" in script


def test_verl_qwen35_compatibility_patch_is_packaged_and_applied() -> None:
    bootstrap = (ROOT / "scripts" / "setup_training.sh").read_text(
        encoding="utf-8"
    )
    patch = (ROOT / "patches" / "verl-qwen35-sft-compat.patch").read_text(
        encoding="utf-8"
    )
    round_robin_patch = (
        ROOT / "patches" / "verl-agent-worker-round-robin.patch"
    ).read_text(encoding="utf-8")
    export_patch = (
        ROOT / "patches" / "verl-export-checkpoint-at-start.patch"
    ).read_text(encoding="utf-8")
    assert "verl-qwen35-sft-compat.patch" in bootstrap
    assert "verl-agent-worker-round-robin.patch" in bootstrap
    assert "verl-export-checkpoint-at-start.patch" in bootstrap
    assert "resolved_messages" in patch
    assert "dummy_is_suffix" in patch
    assert "apply_chat_template(" in patch
    assert "_select_workers" in round_robin_patch
    assert "self._next_worker_index" in round_robin_patch
    assert "VERL_EXPORT_CHECKPOINT_AT_START" in export_patch
    assert "without another optimizer step" in export_patch
