from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

from telecom_post_training.manifest import load_json
from telecom_post_training.settings import ROOT, Settings, validate_deepseek_base_url

STAGE_TO_SPLIT = {
    "smoke": "smoke",
    "train": "train_pool",
    "dev": "dev",
    "test": "test",
}


def _model_id(provider: str, model: str) -> str:
    return model if model.startswith(f"{provider}/") else f"{provider}/{model}"


def _json_arg(value: dict) -> str:
    return json.dumps(value, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preflight(settings: Settings) -> None:
    if not settings.deepseek_api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing")
    if not settings.deepseek_model:
        raise RuntimeError("DEEPSEEK_MODEL is missing")
    validate_deepseek_base_url(settings.deepseek_base_url)
    url = (
        f"http://{os.getenv('VLLM_HOST', '127.0.0.1')}:"
        f"{os.getenv('VLLM_PORT', '8000')}/v1/models"
    )
    try:
        response = httpx.get(url, timeout=5)
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"Local student endpoint is unavailable: {url}") from exc


def build_command(
    settings: Settings,
    *,
    stage: str,
    trials: int,
    max_concurrency: int,
    max_steps: int,
    seed: int,
    output: Path,
) -> list[str]:
    manifest_path = ROOT / "data" / "manifests" / "telecom_manifest.json"
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise TypeError("Invalid manifest")
    split_name = STAGE_TO_SPLIT[stage]
    task_ids = manifest["splits"][split_name]
    tau2_split = {
        "smoke": "small",
        "train": "full",
        "dev": "train",
        "test": "test",
    }[stage]

    agent_args = {
        "temperature": 0.8,
        "max_tokens": int(os.getenv("TAU2_MAX_TOKENS_PER_TURN", "256")),
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    user_args = {
        "temperature": float(os.getenv("DEEPSEEK_USER_TEMPERATURE", "0.7")),
        "max_tokens": 256,
        "seed": int(os.getenv("DEEPSEEK_USER_SEED", "42")),
        "extra_body": {"thinking": {"type": "disabled"}},
    }

    command = [
        str(Path(sys.executable).with_name("tau2")),
        "run",
        "--domain",
        settings.tau2_domain,
        "--task-set-name",
        "telecom",
        "--task-split-name",
        tau2_split,
        "--task-ids",
        *task_ids,
        "--agent",
        "llm_agent",
        "--agent-llm",
        _model_id("openai", settings.vllm_model_alias),
        "--agent-llm-args",
        _json_arg(agent_args),
        "--user",
        "user_simulator",
        "--user-llm",
        _model_id("deepseek", settings.deepseek_model),
        "--user-llm-args",
        _json_arg(user_args),
        "--num-trials",
        str(trials),
        "--max-steps",
        str(max_steps),
        "--max-concurrency",
        str(max_concurrency),
        "--seed",
        str(seed),
        "--save-to",
        str(output.resolve()),
        "--auto-resume",
        "--log-level",
        "INFO",
    ]
    return command


def _write_run_metadata(
    settings: Settings,
    *,
    output: Path,
    stage: str,
    trials: int,
    max_steps: int,
    seed: int,
    command: list[str],
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "stage": stage,
        "trials": trials,
        "max_steps": max_steps,
        "seed": seed,
        "student_model": settings.student_model,
        "student_model_revision": settings.student_model_revision,
        "vllm_model_alias": settings.vllm_model_alias,
        "user_simulator_model": settings.deepseek_model,
        "user_simulator_temperature": float(
            os.getenv("DEEPSEEK_USER_TEMPERATURE", "0.7")
        ),
        "user_simulator_seed": int(os.getenv("DEEPSEEK_USER_SEED", "42")),
        "tau2_commit": os.getenv("TAU2_COMMIT", ""),
        "manifest_sha256": _sha256(
            ROOT / "data" / "manifests" / "telecom_manifest.json"
        ),
        "command_for_audit": [part for part in command if part],
    }
    path = output / "collection_metadata.json"
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def run(args: argparse.Namespace) -> int:
    load_dotenv(ROOT / ".env")
    settings = Settings.from_env()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    command = build_command(
        settings,
        stage=args.stage,
        trials=args.trials,
        max_concurrency=args.max_concurrency,
        max_steps=(
            args.max_steps
            if args.max_steps is not None
            else settings.tau2_rollout_max_turns
        ),
        seed=args.seed,
        output=output,
    )
    if args.dry_run:
        print(" ".join(command))
        return 0

    _preflight(settings)
    metadata_path = _write_run_metadata(
        settings,
        output=output,
        stage=args.stage,
        trials=args.trials,
        max_steps=(
            args.max_steps
            if args.max_steps is not None
            else settings.tau2_rollout_max_turns
        ),
        seed=args.seed,
        command=command,
    )
    print(f"metadata={metadata_path}")
    environment = os.environ.copy()
    environment.update(
        {
            "OPENAI_API_KEY": "EMPTY",
            "OPENAI_BASE_URL": (
                f"http://{os.getenv('VLLM_HOST', '127.0.0.1')}:"
                f"{os.getenv('VLLM_PORT', '8000')}/v1"
            ),
            "DEEPSEEK_API_KEY": settings.deepseek_api_key,
            "DEEPSEEK_API_BASE": settings.deepseek_base_url,
        }
    )
    completed = subprocess.run(command, env=environment, check=False)
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=sorted(STAGE_TO_SPLIT),
        required=True,
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=int(os.getenv("TAU2_ROLLOUT_CONCURRENCY", "2")),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出目录。默认使用 tmp/artifacts/runs/<stage>。",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    args = build_parser().parse_args(argv)
    if not args.output:
        args.output = str(ROOT / "tmp" / "artifacts" / "runs" / args.stage)
    try:
        return run(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
