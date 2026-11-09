import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs" / "profiles" / "96gb.env"
METADATA_TOOL = ROOT / "scripts" / "run_metadata.py"


DEFAULTS = {
    "PROFILE_NAME": "96gb-fast",
    "GPU_MEMORY_GB": "96",
    "SFT_TRAIN_BATCH_SIZE": "8",
    "SFT_MICRO_BATCH_SIZE": "4",
    "SFT_MAX_TOKEN_LEN_PER_GPU": "65536",
    "SFT_MAX_LENGTH": "32768",
    "GRPO_TRAIN_BATCH_SIZE": "8",
    "GRPO_GROUP_SIZE": "8",
    "GRPO_STEPS": "75",
    "GRPO_MINI_BATCH_SIZE": "8",
    "GRPO_MICRO_BATCH_SIZE": "4",
    "GRPO_LOGPROB_MICRO_BATCH_SIZE": "4",
    "GRPO_PARAM_OFFLOAD": "0",
    "GRPO_OPTIMIZER_OFFLOAD": "0",
    "GRPO_VLLM_GPU_MEMORY_UTILIZATION": "0.50",
    "GRPO_VLLM_MAX_NUM_SEQS": "128",
    "AGENT_WORKERS": "16",
    "GRPO_MAX_INFLIGHT_GEN_BATCHES": "1",
    "TAU2_MAX_ACTIVE_ROLLOUTS_PER_WORKER": "2",
    "DEEPSEEK_MAX_CONCURRENCY": "2",
    "DEEPSEEK_MIN_CONCURRENCY": "1",
    "DEEPSEEK_GLOBAL_MAX_CONCURRENCY": "32",
    "DEEPSEEK_FALLBACK_MAX_CONCURRENCY": "4",
    "DEEPSEEK_FALLBACK_MIN_CONCURRENCY": "2",
    "DEEPSEEK_FALLBACK_GLOBAL_MAX_CONCURRENCY": "4",
    "DEEPSEEK_FALLBACK_FUSE_SECONDS": "3600",
    "DEEPSEEK_ADAPTIVE_CONCURRENCY": "1",
    "DEEPSEEK_ADAPTIVE_COOLDOWN_SECONDS": "2",
    "DEEPSEEK_SUCCESSES_TO_RAMP": "8",
    "DEEPSEEK_REQUEST_TIMEOUT_SECONDS": "20",
    "DEEPSEEK_MAX_RETRIES": "5",
    "DEEPSEEK_RETRY_BASE_SECONDS": "0.1",
    "DEEPSEEK_RETRY_JITTER_RATIO": "0.25",
    "DEEPSEEK_RETRY_MAX_SECONDS": "1",
    "DEEPSEEK_TOTAL_REQUEST_DEADLINE_SECONDS": "60",
    "SKIP_SMOKE_VERIFICATION": "0",
    "ALLOW_INSECURE_DEEPSEEK_HTTP": "0",
    "LOGURU_LEVEL": "WARNING",
}


def _profile_env(profile: Path = PROFILE) -> dict[str, str]:
    env = os.environ.copy()
    env.update(DEFAULTS)
    env["PROFILE_FILE"] = str(profile)
    return env


def _run_metadata(
    *args: str,
    env: dict[str, str],
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(METADATA_TOOL), *args],
        env=env,
        check=check,
        capture_output=True,
        text=True,
    )


def test_96gb_profile_defaults_and_override() -> None:
    result = _run_metadata("validate", env=_profile_env())
    config = json.loads(result.stdout)
    assert config["PROFILE_NAME"] == "96gb-fast"
    assert config["SFT_TRAIN_BATCH_SIZE"] == 8
    assert config["SFT_MICRO_BATCH_SIZE"] == 4
    assert config["SFT_MAX_TOKEN_LEN_PER_GPU"] == 65536
    assert config["GRPO_TRAIN_BATCH_SIZE"] == 8
    assert config["GRPO_GROUP_SIZE"] == 8
    assert config["GRPO_STEPS"] == 75
    assert config["GRPO_MICRO_BATCH_SIZE"] == 4
    assert config["GRPO_LOGPROB_MICRO_BATCH_SIZE"] == 4
    assert config["GRPO_PARAM_OFFLOAD"] is False
    assert config["GRPO_OPTIMIZER_OFFLOAD"] is False
    assert config["GRPO_VLLM_GPU_MEMORY_UTILIZATION"] == 0.5
    assert config["GRPO_VLLM_MAX_NUM_SEQS"] == 128
    assert config["AGENT_WORKERS"] == 16
    assert config["GRPO_MAX_INFLIGHT_GEN_BATCHES"] == 1
    assert config["TAU2_MAX_ACTIVE_ROLLOUTS_PER_WORKER"] == 2
    assert config["DEEPSEEK_MAX_CONCURRENCY"] == 2
    assert config["DEEPSEEK_MIN_CONCURRENCY"] == 1
    assert config["DEEPSEEK_GLOBAL_MAX_CONCURRENCY"] == 32
    assert config["DEEPSEEK_FALLBACK_MAX_CONCURRENCY"] == 4
    assert config["DEEPSEEK_FALLBACK_MIN_CONCURRENCY"] == 2
    assert config["DEEPSEEK_FALLBACK_GLOBAL_MAX_CONCURRENCY"] == 4
    assert config["DEEPSEEK_ADAPTIVE_CONCURRENCY"] is True
    assert config["DEEPSEEK_ADAPTIVE_COOLDOWN_SECONDS"] == 2
    assert config["DEEPSEEK_SUCCESSES_TO_RAMP"] == 8
    assert config["DEEPSEEK_RETRY_JITTER_RATIO"] == 0.25
    assert config["DEEPSEEK_RETRY_MAX_SECONDS"] == 1
    assert config["DEEPSEEK_TOTAL_REQUEST_DEADLINE_SECONDS"] == 60
    assert config["SKIP_SMOKE_VERIFICATION"] is False
    assert config["ALLOW_INSECURE_DEEPSEEK_HTTP"] is False
    assert config["LOGURU_LEVEL"] == "WARNING"

    overridden = _profile_env()
    overridden["SFT_MICRO_BATCH_SIZE"] = "2"
    overridden["AGENT_WORKERS"] = "8"
    overridden["GRPO_MAX_INFLIGHT_GEN_BATCHES"] = "2"
    overridden["TAU2_MAX_ACTIVE_ROLLOUTS_PER_WORKER"] = "1"
    config = json.loads(
        _run_metadata("validate", env=overridden).stdout
    )
    assert config["SFT_MICRO_BATCH_SIZE"] == 2
    assert config["AGENT_WORKERS"] == 8
    assert config["GRPO_MAX_INFLIGHT_GEN_BATCHES"] == 2
    assert config["TAU2_MAX_ACTIVE_ROLLOUTS_PER_WORKER"] == 1


def test_shell_profile_preserves_exported_overrides(tmp_path: Path) -> None:
    script = tmp_path / "check_profile.sh"
    script.write_text(
        """set -euo pipefail
export PROFILE_FILE=configs/profiles/96gb.env
export SFT_MICRO_BATCH_SIZE=2
export AGENT_WORKERS=8
source scripts/lib_profile.sh
load_training_profile "$PWD"
printf "%s %s %s\\n" "$SFT_TRAIN_BATCH_SIZE" "$SFT_MICRO_BATCH_SIZE" "$AGENT_WORKERS"
""",
        encoding="utf-8",
        newline="\n",
    )
    command = ["bash", str(script)]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.stdout.strip() == "8 2 8"


def test_invalid_profile_fails() -> None:
    env = _profile_env()
    env["GRPO_TRAIN_BATCH_SIZE"] = "16"
    with pytest.raises(subprocess.CalledProcessError) as exc:
        _run_metadata("validate", env=env)
    assert "at most 8" in exc.value.stderr


def test_invalid_grpo_inflight_batch_limit_fails() -> None:
    env = _profile_env()
    env["GRPO_MAX_INFLIGHT_GEN_BATCHES"] = "0"
    with pytest.raises(subprocess.CalledProcessError) as exc:
        _run_metadata("validate", env=env)
    assert "max inflight generation batches" in exc.value.stderr


def test_invalid_rollout_concurrency_limit_fails() -> None:
    env = _profile_env()
    env["TAU2_MAX_ACTIVE_ROLLOUTS_PER_WORKER"] = "0"
    with pytest.raises(subprocess.CalledProcessError) as exc:
        _run_metadata("validate", env=env)
    assert "active rollouts per worker" in exc.value.stderr


def test_invalid_deepseek_concurrency_bounds_fail() -> None:
    env = _profile_env()
    env["DEEPSEEK_MIN_CONCURRENCY"] = "17"
    env["DEEPSEEK_MAX_CONCURRENCY"] = "16"
    with pytest.raises(subprocess.CalledProcessError) as exc:
        _run_metadata("validate", env=env)
    assert "minimum concurrency" in exc.value.stderr


def test_invalid_insecure_http_flag_fails() -> None:
    env = _profile_env()
    env["ALLOW_INSECURE_DEEPSEEK_HTTP"] = "sometimes"
    with pytest.raises(subprocess.CalledProcessError) as exc:
        _run_metadata("validate", env=env)
    assert "Invalid boolean" in exc.value.stderr


def test_final_config_matches_96gb_profile() -> None:
    config = yaml.safe_load(
        (ROOT / "configs" / "final_recommended.yaml").read_text(encoding="utf-8")
    )
    assert config["hardware"]["gpu_memory_gb"] == 96
    assert config["hardware"]["profile"] == "configs/profiles/96gb.env"
    assert config["hardware"]["profile_name"] == "96gb-fast"
    assert (
        config["runtime_compatibility"]["verl_round_robin_patch"]
        == "patches/verl-agent-worker-round-robin.patch"
    )
    assert (
        config["runtime_compatibility"][
            "verl_round_robin_patch_sha256"
        ]
        == hashlib.sha256(
            (ROOT / "patches" / "verl-agent-worker-round-robin.patch").read_bytes()
        ).hexdigest()
    )
    assert (
        config["runtime_compatibility"]["verl_export_patch"]
        == "patches/verl-export-checkpoint-at-start.patch"
    )
    assert (
        config["runtime_compatibility"]["verl_export_patch_sha256"]
        == hashlib.sha256(
            (ROOT / "patches" / "verl-export-checkpoint-at-start.patch").read_bytes()
        ).hexdigest()
    )
    assert config["smoke_verification"]["default"] == "required_marker"
    assert config["smoke_verification"]["current_run"] == "skipped_by_user"
    assert config["sft"]["train_batch_size"] == 8
    assert config["sft"]["micro_batch_size_per_gpu"] == 4
    assert config["sft"]["max_token_len_per_gpu"] == 65536
    assert config["sft"]["max_length"] == 32768
    assert config["grpo"]["train_batch_prompts"] == 8
    assert config["grpo"]["group_size"] == 8
    assert config["grpo"]["optimizer_steps"] == 75
    assert config["grpo"]["mini_batch_size"] == 8
    assert config["grpo"]["micro_batch_size_per_gpu"] == 4
    assert config["grpo"]["logprob_micro_batch_size_per_gpu"] == 4
    assert config["grpo"]["param_offload"] is False
    assert config["grpo"]["optimizer_offload"] is False
    assert config["grpo"]["vllm_gpu_memory_utilization"] == 0.5
    assert config["grpo"]["vllm_max_num_seqs"] == 128
    assert config["grpo"]["agent_workers"] == 16
    assert config["grpo"]["max_inflight_gen_batches"] == 1
    assert config["grpo"]["max_inflight_trajectories"] == 64
    assert config["grpo"]["max_active_rollouts_per_worker"] == 2
    assert config["grpo"]["max_simultaneous_rollouts"] == 16
    assert config["remote_models"]["client"]["max_concurrency_per_process"] == 2
    assert config["remote_models"]["client"]["min_concurrency_per_process"] == 1
    assert config["remote_models"]["client"]["gate_scope"] == "process"
    assert (
        config["remote_models"]["client"]["transport"]
        == "configured_endpoint"
    )
    assert (
        config["remote_models"]["client"]["primary_endpoint"]
        == "${DEEPSEEK_BASE_URL}"
    )
    assert config["remote_models"]["client"]["global_concurrency_limit"] == 32
    assert (
        config["remote_models"]["client"]["fallback"]["global_limit"] == 4
    )
    assert config["remote_models"]["client"]["adaptive_concurrency"] is True
    assert config["remote_models"]["client"]["request_timeout_seconds"] == 20
    assert config["remote_models"]["client"]["max_retries"] == 5
    assert config["remote_models"]["client"]["retry_base_seconds"] == 0.1
    assert config["remote_models"]["client"]["retry_jitter_ratio"] == 0.25
    assert config["remote_models"]["client"]["retry_max_seconds"] == 1
    assert (
        config["remote_models"]["client"]["total_request_deadline_seconds"]
        == 60
    )
    assert config["remote_models"]["logging"]["tau2_level"] == "WARNING"


def test_smoke_marker_binds_config_and_data_hashes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "configs" / "profiles").mkdir(parents=True)
    (repo / "data").mkdir()
    (repo / "model").mkdir()
    (repo / "configs" / "profiles" / "96gb.env").write_text(
        PROFILE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (repo / ".gitignore").write_text("data/\nmodel/\n", encoding="utf-8")
    data = repo / "data" / "train.parquet"
    data.write_bytes(b"data-v1")
    (repo / "model" / "config.json").write_text("{}", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "add", ".gitignore", "configs/profiles/96gb.env"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "init"], check=True)

    snapshot = tmp_path / "snapshot.json"
    marker = tmp_path / "smoke.ok"
    env = _profile_env(repo / "configs" / "profiles" / "96gb.env")
    _run_metadata(
        "snapshot",
        "--root",
        str(repo),
        "--phase",
        "sft-smoke",
        "--arm",
        "uniform",
        "--data-file",
        str(data),
        "--model-path",
        str(repo / "model"),
        "--train-file",
        str(data),
        "--val-file",
        "null",
        "--steps",
        "1",
        "--nproc-per-node",
        "1",
        "--output",
        str(snapshot),
        env=env,
    )
    _run_metadata(
        "mark-smoke",
        "--snapshot",
        str(snapshot),
        "--output",
        str(marker),
        env=env,
    )
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["smoke_passed"] is True
    assert payload["smoke_acceptance"] == "smoke_run"
    assert len(payload["git_commit"]) == 40
    assert len(payload["profile_sha256"]) == 64
    assert len(payload["effective_config_hash"]) == 64
    assert len(next(iter(payload["data_sha256"].values()))) == 64

    skipped_snapshot = tmp_path / "skipped_snapshot.json"
    skipped_env = dict(env)
    skipped_env["SKIP_SMOKE_VERIFICATION"] = "1"
    _run_metadata(
        "snapshot",
        "--root",
        str(repo),
        "--phase",
        "formal",
        "--arm",
        "uniform",
        "--data-file",
        str(data),
        "--model-path",
        str(repo / "model"),
        "--train-file",
        str(data),
        "--val-file",
        "null",
        "--steps",
        "100",
        "--nproc-per-node",
        "1",
        "--output",
        str(skipped_snapshot),
        env=skipped_env,
    )
    skipped = json.loads(skipped_snapshot.read_text(encoding="utf-8"))
    assert skipped["smoke_acceptance"] == "skipped_by_user"
    assert skipped["effective_config"]["SKIP_SMOKE_VERIFICATION"] is True
    assert skipped["git_commit"] == payload["git_commit"]
    assert skipped["data_sha256"] == payload["data_sha256"]
    _run_metadata(
        "verify-smoke",
        "--root",
        str(repo),
        "--phase",
        "sft-smoke",
        "--marker",
        str(marker),
        "--require-clean",
        env=env,
    )

    dirty = repo / "dirty.txt"
    dirty.write_text("uncommitted\n", encoding="utf-8")
    with pytest.raises(subprocess.CalledProcessError) as exc:
        _run_metadata(
            "verify-smoke",
            "--root",
            str(repo),
            "--phase",
            "sft-smoke",
            "--marker",
            str(marker),
            "--require-clean",
            env=env,
        )
    assert "not clean" in exc.value.stderr
    dirty.unlink()

    changed_env = dict(env)
    changed_env["GRPO_MAX_INFLIGHT_GEN_BATCHES"] = "2"
    with pytest.raises(subprocess.CalledProcessError) as exc:
        _run_metadata(
            "verify-smoke",
            "--root",
            str(repo),
            "--phase",
            "sft-smoke",
            "--marker",
            str(marker),
            "--require-clean",
            env=changed_env,
        )
    assert "configuration hash is stale" in exc.value.stderr

    changed_route_env = dict(env)
    changed_route_env["DEEPSEEK_FALLBACK_BASE_URL"] = (
        "https://fallback.example.com/v1"
    )
    changed_route_env["DEEPSEEK_FALLBACK_API_KEY"] = "route-test-secret"
    with pytest.raises(subprocess.CalledProcessError) as exc:
        _run_metadata(
            "verify-smoke",
            "--root",
            str(repo),
            "--phase",
            "sft-smoke",
            "--marker",
            str(marker),
            "--require-clean",
            env=changed_route_env,
        )
    assert "configuration hash is stale" in exc.value.stderr

    data.write_bytes(b"data-v2")
    with pytest.raises(subprocess.CalledProcessError) as exc:
        _run_metadata(
            "verify-smoke",
            "--root",
            str(repo),
            "--phase",
            "sft-smoke",
            "--marker",
            str(marker),
            env=env,
        )
    assert "data hash is stale" in exc.value.stderr


def test_both_training_scripts_load_the_same_shared_profile() -> None:
    sft = (ROOT / "scripts" / "train_sft.sh").read_text(encoding="utf-8")
    grpo = (ROOT / "scripts" / "train_grpo.sh").read_text(encoding="utf-8")
    for script in (sft, grpo):
        assert 'source "$ROOT/scripts/lib_profile.sh"' in script
        assert 'load_training_profile "$ROOT"' in script
    for name in DEFAULTS:
        if name.startswith("SFT_"):
            assert f"${name}" in sft
        elif name.startswith("GRPO_"):
            if "OFFLOAD" in name:
                assert name in grpo
            else:
                assert f"${name}" in grpo


def test_skip_smoke_override_is_explicit_and_logged() -> None:
    for name in ("train_sft.sh", "train_grpo.sh"):
        script = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert 'SKIP_SMOKE_VERIFICATION="${SKIP_SMOKE_VERIFICATION:-0}"' in script
        assert '[[ "$SKIP_SMOKE_VERIFICATION" == "1" ]]' in script
        assert "SMOKE_ACCEPTANCE=skipped_by_user" in script
        assert "RUN_ACCEPTANCE " in script
