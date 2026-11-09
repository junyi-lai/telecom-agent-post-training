from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _boolean(raw: str) -> bool:
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {raw}")


def _log_level(raw: str) -> str:
    return raw.strip().upper()


PROFILE_KEYS = {
    "PROFILE_NAME": str,
    "GPU_MEMORY_GB": int,
    "SFT_TRAIN_BATCH_SIZE": int,
    "SFT_MICRO_BATCH_SIZE": int,
    "SFT_MAX_TOKEN_LEN_PER_GPU": int,
    "SFT_MAX_LENGTH": int,
    "GRPO_TRAIN_BATCH_SIZE": int,
    "GRPO_GROUP_SIZE": int,
    "GRPO_STEPS": int,
    "GRPO_MINI_BATCH_SIZE": int,
    "GRPO_MICRO_BATCH_SIZE": int,
    "GRPO_LOGPROB_MICRO_BATCH_SIZE": int,
    "GRPO_PARAM_OFFLOAD": _boolean,
    "GRPO_OPTIMIZER_OFFLOAD": _boolean,
    "GRPO_VLLM_GPU_MEMORY_UTILIZATION": float,
    "GRPO_VLLM_MAX_NUM_SEQS": int,
    "AGENT_WORKERS": int,
    "GRPO_MAX_INFLIGHT_GEN_BATCHES": int,
    "TAU2_MAX_ACTIVE_ROLLOUTS_PER_WORKER": int,
    "DEEPSEEK_MAX_CONCURRENCY": int,
    "DEEPSEEK_MIN_CONCURRENCY": int,
    "DEEPSEEK_GLOBAL_MAX_CONCURRENCY": int,
    "DEEPSEEK_FALLBACK_MAX_CONCURRENCY": int,
    "DEEPSEEK_FALLBACK_MIN_CONCURRENCY": int,
    "DEEPSEEK_FALLBACK_GLOBAL_MAX_CONCURRENCY": int,
    "DEEPSEEK_FALLBACK_FUSE_SECONDS": float,
    "DEEPSEEK_ADAPTIVE_CONCURRENCY": _boolean,
    "DEEPSEEK_ADAPTIVE_COOLDOWN_SECONDS": float,
    "DEEPSEEK_SUCCESSES_TO_RAMP": int,
    "DEEPSEEK_REQUEST_TIMEOUT_SECONDS": float,
    "DEEPSEEK_MAX_RETRIES": int,
    "DEEPSEEK_RETRY_BASE_SECONDS": float,
    "DEEPSEEK_RETRY_JITTER_RATIO": float,
    "DEEPSEEK_RETRY_MAX_SECONDS": float,
    "DEEPSEEK_TOTAL_REQUEST_DEADLINE_SECONDS": float,
    "SKIP_SMOKE_VERIFICATION": _boolean,
    "ALLOW_INSECURE_DEEPSEEK_HTTP": _boolean,
    "LOGURU_LEVEL": _log_level,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _effective_config() -> dict[str, Any]:
    config = {}
    missing = []
    for name, converter in PROFILE_KEYS.items():
        raw = os.getenv(name)
        if raw is None or raw == "":
            missing.append(name)
            continue
        config[name] = converter(raw)
    if missing:
        raise ValueError(f"Missing profile variables: {', '.join(missing)}")
    return config


def _routing_config() -> dict[str, Any]:
    primary_base_url = os.getenv(
        "DEEPSEEK_BASE_URL",
        "https://api.deepseek.com/v1",
    ).strip()
    fallback_base_url = os.getenv("DEEPSEEK_FALLBACK_BASE_URL", "").strip()
    fallback_key_present = bool(os.getenv("DEEPSEEK_FALLBACK_API_KEY", "").strip())
    return {
        "primary_base_url": primary_base_url,
        "primary_model": os.getenv("DEEPSEEK_MODEL", "deepseek-flash").strip(),
        "primary_request_model": (
            os.getenv("DEEPSEEK_REQUEST_MODEL", "").strip()
            or os.getenv("DEEPSEEK_MODEL", "deepseek-flash").strip()
        ),
        "fallback_base_url": fallback_base_url,
        "fallback_model": os.getenv(
            "DEEPSEEK_FALLBACK_MODEL",
            "deepseek-flash",
        ).strip(),
        "fallback_enabled": bool(fallback_base_url and fallback_key_present),
        "fallback_api_key_present": fallback_key_present,
    }


def _config_hash(config: dict[str, Any]) -> str:
    return _canonical_hash(
        {
            "profile": config,
            "routing": _routing_config(),
        }
    )


def _validate(config: dict[str, Any]) -> None:
    if config["GPU_MEMORY_GB"] < 48:
        raise ValueError("The training profile requires at least 48 GB GPU memory")
    if not 1 <= config["SFT_MICRO_BATCH_SIZE"] <= config["SFT_TRAIN_BATCH_SIZE"]:
        raise ValueError("SFT micro batch must be between 1 and train batch")
    if config["SFT_MAX_TOKEN_LEN_PER_GPU"] < config["SFT_MAX_LENGTH"]:
        raise ValueError("SFT max token per GPU must cover one sample")
    if config["GRPO_TRAIN_BATCH_SIZE"] > 8:
        raise ValueError("GRPO train batch must remain at most 8 prompts")
    if not 1 <= config["GRPO_MICRO_BATCH_SIZE"] <= config["GRPO_MINI_BATCH_SIZE"]:
        raise ValueError("GRPO micro batch must not exceed mini batch")
    if not (
        1
        <= config["GRPO_LOGPROB_MICRO_BATCH_SIZE"]
        <= config["GRPO_MINI_BATCH_SIZE"]
    ):
        raise ValueError("GRPO logprob micro batch must not exceed mini batch")
    if not 1 <= config["AGENT_WORKERS"] <= 128:
        raise ValueError("Agent workers must be between 1 and 128")
    if not 1 <= config["GRPO_MAX_INFLIGHT_GEN_BATCHES"] <= 2:
        raise ValueError(
            "GRPO max inflight generation batches must be between 1 and 2"
        )
    if not 1 <= config["TAU2_MAX_ACTIVE_ROLLOUTS_PER_WORKER"] <= 8:
        raise ValueError(
            "Tau2 active rollouts per worker must be between 1 and 8"
        )
    if not 1 <= config["DEEPSEEK_MAX_CONCURRENCY"] <= 64:
        raise ValueError("DeepSeek concurrency must be between 1 and 64")
    if not (
        1
        <= config["DEEPSEEK_MIN_CONCURRENCY"]
        <= config["DEEPSEEK_MAX_CONCURRENCY"]
    ):
        raise ValueError(
            "DeepSeek minimum concurrency must not exceed the maximum"
        )
    if not 1 <= config["DEEPSEEK_GLOBAL_MAX_CONCURRENCY"] <= 256:
        raise ValueError(
            "DeepSeek global concurrency must be between 1 and 256"
        )
    if not (
        1
        <= config["DEEPSEEK_FALLBACK_MIN_CONCURRENCY"]
        <= config["DEEPSEEK_FALLBACK_MAX_CONCURRENCY"]
        <= 64
    ):
        raise ValueError("DeepSeek fallback concurrency bounds are invalid")
    if not 1 <= config["DEEPSEEK_FALLBACK_GLOBAL_MAX_CONCURRENCY"] <= 64:
        raise ValueError(
            "DeepSeek fallback global concurrency must be between 1 and 64"
        )
    if config["DEEPSEEK_ADAPTIVE_COOLDOWN_SECONDS"] < 0:
        raise ValueError(
            "DeepSeek adaptive concurrency cooldown must be nonnegative"
        )
    if config["DEEPSEEK_SUCCESSES_TO_RAMP"] < 1:
        raise ValueError("DeepSeek successes-to-ramp must be positive")
    if not 0 <= config["DEEPSEEK_RETRY_JITTER_RATIO"] <= 2:
        raise ValueError("DeepSeek retry jitter ratio must be in [0, 2]")
    if config["DEEPSEEK_RETRY_MAX_SECONDS"] < 0:
        raise ValueError("DeepSeek retry max seconds must be nonnegative")
    if config["DEEPSEEK_TOTAL_REQUEST_DEADLINE_SECONDS"] <= 0:
        raise ValueError("DeepSeek total request deadline must be positive")
    if config["LOGURU_LEVEL"] not in {
        "TRACE",
        "DEBUG",
        "INFO",
        "SUCCESS",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }:
        raise ValueError("Unsupported LOGURU_LEVEL")
    if not 0 < config["GRPO_VLLM_GPU_MEMORY_UTILIZATION"] <= 0.95:
        raise ValueError("vLLM GPU memory utilization must be in (0, 0.95]")
    if config["GRPO_VLLM_MAX_NUM_SEQS"] < 1:
        raise ValueError("vLLM max num seqs must be positive")


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args],
        text=True,
    ).strip()


def _relative(root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError as exc:
        raise ValueError(f"Path is outside project root: {path}") from exc


def _snapshot(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    config = _effective_config()
    _validate(config)
    profile_path = Path(os.environ["PROFILE_FILE"])
    if not profile_path.is_absolute():
        profile_path = root / profile_path
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile file does not exist: {profile_path}")

    data_hashes = {}
    for raw_path in args.data_file:
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise FileNotFoundError(f"Data file does not exist: {path}")
        data_hashes[_relative(root, path)] = _sha256(path)

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = root / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    train_file = Path(args.train_file)
    if not train_file.is_absolute():
        train_file = root / train_file
    validation_value = args.val_file
    validation_file = None
    if validation_value != "null":
        validation_path = Path(validation_value)
        if not validation_path.is_absolute():
            validation_path = root / validation_path
        validation_file = _relative(root, validation_path)
    skip_smoke = config["SKIP_SMOKE_VERIFICATION"]
    if args.phase.endswith("-smoke"):
        smoke_acceptance = "smoke_run"
    elif skip_smoke:
        smoke_acceptance = "skipped_by_user"
    else:
        smoke_acceptance = "verified_marker"

    return {
        "created_at": datetime.now(UTC).isoformat(),
        "phase": args.phase,
        "arm": args.arm,
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_dirty": bool(_git(root, "status", "--porcelain")),
        "profile_file": _relative(root, profile_path),
        "profile_sha256": _sha256(profile_path),
        "effective_config": config,
        "routing_config": _routing_config(),
        "effective_config_hash": _config_hash(config),
        "data_sha256": data_hashes,
        "model_path": _relative(root, model_path),
        "smoke_acceptance": smoke_acceptance,
        "model_revision": os.getenv("STUDENT_MODEL_REVISION", ""),
        "run_settings": {
            "steps": args.steps,
            "nproc_per_node": args.nproc_per_node,
            "train_file": _relative(root, train_file),
            "validation_file": validation_file,
        },
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _snapshot_command(args: argparse.Namespace) -> int:
    snapshot = _snapshot(args)
    _write_json(args.output, snapshot)
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    return 0


def _mark_command(args: argparse.Namespace) -> int:
    marker = json.loads(args.snapshot.read_text(encoding="utf-8"))
    marker["smoke_passed"] = True
    marker["smoke_marked_at"] = datetime.now(UTC).isoformat()
    _write_json(args.output, marker)
    return 0


def _verify_command(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    marker = json.loads(args.marker.read_text(encoding="utf-8"))
    if not marker.get("smoke_passed"):
        raise ValueError("Smoke marker is not marked as passed")
    if marker.get("phase") != args.phase:
        raise ValueError(
            f"Smoke phase mismatch: expected {args.phase}, found {marker.get('phase')}"
        )
    if args.require_clean and _git(root, "status", "--porcelain"):
        raise ValueError("Repository is not clean")

    current_commit = _git(root, "rev-parse", "HEAD")
    if marker.get("git_commit") != current_commit:
        raise ValueError("Smoke marker does not match the current Git commit")

    config = _effective_config()
    _validate(config)
    profile_path = Path(os.environ["PROFILE_FILE"])
    if not profile_path.is_absolute():
        profile_path = root / profile_path
    if marker.get("profile_sha256") != _sha256(profile_path):
        raise ValueError("Smoke marker profile hash is stale")
    if marker.get("effective_config_hash") != _config_hash(config):
        raise ValueError("Smoke marker effective configuration hash is stale")

    marker_data = marker.get("data_sha256") or {}
    for relative_path, expected_hash in marker_data.items():
        path = root / relative_path
        if not path.exists() or _sha256(path) != expected_hash:
            raise ValueError(f"Smoke marker data hash is stale: {relative_path}")
    print(
        "SMOKE_MARKER_OK "
        f"phase={args.phase} commit={current_commit} "
        f"config_hash={marker['effective_config_hash']}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.set_defaults(func=None)

    snapshot_parser = subparsers.add_parser("snapshot")
    snapshot_parser.add_argument("--root", type=Path, required=True)
    snapshot_parser.add_argument("--phase", required=True)
    snapshot_parser.add_argument("--arm", required=True)
    snapshot_parser.add_argument("--data-file", action="append", default=[])
    snapshot_parser.add_argument("--model-path", required=True)
    snapshot_parser.add_argument("--train-file", required=True)
    snapshot_parser.add_argument("--val-file", required=True)
    snapshot_parser.add_argument("--steps", required=True)
    snapshot_parser.add_argument("--nproc-per-node", type=int, required=True)
    snapshot_parser.add_argument("--output", type=Path, required=True)
    snapshot_parser.set_defaults(func=_snapshot_command)

    mark_parser = subparsers.add_parser("mark-smoke")
    mark_parser.add_argument("--snapshot", type=Path, required=True)
    mark_parser.add_argument("--output", type=Path, required=True)
    mark_parser.set_defaults(func=_mark_command)

    verify_parser = subparsers.add_parser("verify-smoke")
    verify_parser.add_argument("--root", type=Path, required=True)
    verify_parser.add_argument("--phase", required=True)
    verify_parser.add_argument("--marker", type=Path, required=True)
    verify_parser.add_argument("--require-clean", action="store_true")
    verify_parser.set_defaults(func=_verify_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        config = _effective_config()
        _validate(config)
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
