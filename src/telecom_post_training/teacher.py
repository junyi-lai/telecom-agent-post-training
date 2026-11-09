from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from telecom_post_training.settings import ROOT, Settings, validate_deepseek_base_url


def _model_id(provider: str, model: str) -> str:
    return model if model.startswith(f"{provider}/") else f"{provider}/{model}"


def _load_failure_task_ids(path: Path) -> list[str]:
    task_ids: list[str] = []
    if not path.exists():
        raise FileNotFoundError(f"Missing normalized baseline file: {path}")
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("strict_success") and record.get("task_id"):
                task_ids.append(record["task_id"])
    return list(dict.fromkeys(task_ids))


def build_command(
    settings: Settings,
    *,
    task_ids: list[str],
    trials: int,
    output: Path,
) -> list[str]:
    teacher_args = {
        "temperature": float(os.getenv("DEEPSEEK_TEACHER_TEMPERATURE", "0.2")),
        "max_tokens": 1024,
        "seed": int(os.getenv("DEEPSEEK_TEACHER_SEED", "43")),
        "extra_body": {"thinking": {"type": "enabled"}},
    }
    user_args = {
        "temperature": float(os.getenv("DEEPSEEK_USER_TEMPERATURE", "0.7")),
        "max_tokens": 256,
        "seed": int(os.getenv("DEEPSEEK_USER_SEED", "42")),
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    teacher_model = _model_id("deepseek", settings.deepseek_teacher_model)
    user_model = _model_id("deepseek", settings.deepseek_model)
    return [
        str(Path(sys.executable).with_name("tau2")),
        "run",
        "--domain",
        settings.tau2_domain,
        "--task-set-name",
        "telecom",
        "--task-split-name",
        "full",
        "--task-ids",
        *task_ids,
        "--agent",
        "llm_agent_gt",
        "--agent-llm",
        teacher_model,
        "--agent-llm-args",
        json.dumps(teacher_args, separators=(",", ":")),
        "--user",
        "user_simulator",
        "--user-llm",
        user_model,
        "--user-llm-args",
        json.dumps(user_args, separators=(",", ":")),
        "--num-trials",
        str(trials),
        "--max-steps",
        os.getenv("TAU2_TEACHER_MAX_STEPS", "60"),
        "--max-concurrency",
        str(settings.tau2_rollout_concurrency),
        "--seed",
        "20261103",
        "--save-to",
        str(output.resolve()),
        "--auto-resume",
        "--log-level",
        "INFO",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-input",
        default=str(ROOT / "data" / "processed" / "train.jsonl"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "tmp" / "artifacts" / "runs" / "teacher"),
    )
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--trials", type=int, default=2)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    load_dotenv(ROOT / ".env")
    settings = Settings.from_env()
    baseline = Path(args.baseline_input)
    if not baseline.is_absolute():
        baseline = ROOT / baseline
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    try:
        task_ids = _load_failure_task_ids(baseline)
        if args.max_tasks:
            task_ids = task_ids[: args.max_tasks]
        if not task_ids:
            raise RuntimeError("No failed task ids found in the baseline")
        command = build_command(
            settings,
            task_ids=task_ids,
            trials=args.trials,
            output=output,
        )
        if args.dry_run:
            print(" ".join(command))
            return 0
        validate_deepseek_base_url(settings.deepseek_base_url)
        if not settings.deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is missing")
        if not settings.deepseek_teacher_model:
            raise RuntimeError("DEEPSEEK_TEACHER_MODEL is missing")
        environment = os.environ.copy()
        environment["DEEPSEEK_API_KEY"] = settings.deepseek_api_key
        environment["DEEPSEEK_API_BASE"] = settings.deepseek_base_url
        output.mkdir(parents=True, exist_ok=True)
        (output / "teacher_task_ids.json").write_text(
            json.dumps(task_ids, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return subprocess.run(command, env=environment, check=False).returncode
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
