from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from telecom_post_training.analysis import (
    load_jsonl,
    paired_task_bootstrap,
    render_markdown,
    render_paired_comparison,
    summarize,
)
from telecom_post_training.context import write_runtime_context
from telecom_post_training.dataset import build_datasets
from telecom_post_training.manifest import build_manifest, load_json, write_manifest
from telecom_post_training.normalization import (
    iter_simulations,
    normalize_simulation,
    write_jsonl,
)
from telecom_post_training.rl_dataset import build_rl_prompts
from telecom_post_training.settings import ROOT, Settings


def _settings() -> Settings:
    load_dotenv(ROOT / ".env")
    return Settings.from_env()


def _cmd_manifest(args: argparse.Namespace) -> int:
    settings = _settings()
    manifest = build_manifest(settings.tau2_repo, seed=args.seed)
    destination = (
        Path(args.output)
        if args.output
        else ROOT / "data" / "manifests" / "telecom_manifest.json"
    )
    if not destination.is_absolute():
        destination = ROOT / destination
    write_manifest(manifest, destination)
    print(json.dumps(manifest["counts"], indent=2))
    print(f"manifest={destination}")
    return 0


def _cmd_normalize(args: argparse.Namespace) -> int:
    settings = _settings()
    source = Path(args.input)
    if not source.is_absolute():
        source = ROOT / source
    destination = Path(args.output)
    if not destination.is_absolute():
        destination = ROOT / destination
    simulations = iter_simulations(source)
    records = [
        normalize_simulation(
            simulation,
            split=args.split,
            model_revision=settings.student_model_revision,
            user_simulator_model=settings.deepseek_model,
            source_role=(
                "offline_teacher" if args.split == "teacher" else "student"
            ),
            teacher_model=(
                settings.deepseek_teacher_model
                if args.split == "teacher"
                else None
            ),
        )
        for simulation in simulations
    ]
    write_jsonl(records, destination)
    success = sum(record["strict_success"] for record in records)
    print(
        json.dumps(
            {
                "records": len(records),
                "success": success,
                "success_rate": round(success / len(records), 4) if records else 0,
                "output": str(destination),
            },
            indent=2,
        )
    )
    return 0


def _cmd_context(args: argparse.Namespace) -> int:
    destination = Path(args.output)
    if not destination.is_absolute():
        destination = ROOT / destination
    context = write_runtime_context(destination)
    print(
        json.dumps(
            {
                "system_prompt_chars": len(context["system_prompt"]),
                "tool_count": len(context["tools"]),
                "sha256": context["sha256"],
                "output": str(destination),
            },
            indent=2,
        )
    )
    return 0


def _cmd_build_dataset(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    settings = _settings()
    sources = []
    for item in args.input:
        source = Path(item)
        if not source.is_absolute():
            source = ROOT / source
        sources.append(source)
    failure_source = None
    if args.failure_source:
        failure_source = Path(args.failure_source)
        if not failure_source.is_absolute():
            failure_source = ROOT / failure_source
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    model_source = (
        str(settings.student_model_path)
        if settings.student_model_path.exists()
        else settings.student_model
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        revision=None
        if settings.student_model_path.exists()
        else settings.student_model_revision,
    )
    manifest = load_json(ROOT / "data" / "manifests" / "telecom_manifest.json")
    if not isinstance(manifest, dict):
        raise TypeError("Invalid manifest")
    summary = build_datasets(
        sources,
        output_dir,
        tokenizer=tokenizer,
        target_tokens=args.target_tokens,
        seed=args.seed,
        failure_source=failure_source,
        required_task_ids=manifest["splits"]["train_pool"],
    )
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_build_rl(args: argparse.Namespace) -> int:
    manifest = load_json(ROOT / "data" / "manifests" / "telecom_manifest.json")
    source = Path(args.baseline_input)
    if not source.is_absolute():
        source = ROOT / source
    records = []
    if source.exists():
        with source.open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    summary = build_rl_prompts(
        manifest,
        write_runtime_context(
            ROOT / "tmp" / "data" / "processed" / "runtime_context.json"
        ),
        records,
        output_dir=output_dir,
        seed=args.seed,
        task_split=args.task_split,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    source = Path(args.input)
    if not source.is_absolute():
        source = ROOT / source
    records = load_jsonl(source)
    summary = summarize(records)
    destination = Path(args.output)
    if not destination.is_absolute():
        destination = ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_compare_tasks(args: argparse.Namespace) -> int:
    left = Path(args.left)
    right = Path(args.right)
    destination = Path(args.output)
    if not left.is_absolute():
        left = ROOT / left
    if not right.is_absolute():
        right = ROOT / right
    if not destination.is_absolute():
        destination = ROOT / destination
    result = paired_task_bootstrap(
        load_jsonl(left),
        load_jsonl(right),
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_paired_comparison(result), encoding="utf-8")
    json_destination = destination.with_suffix(".json")
    json_destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="telecom-post-training")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--seed", type=int, default=20261103)
    manifest_parser.add_argument("--output")
    manifest_parser.set_defaults(func=_cmd_manifest)

    normalize_parser = subparsers.add_parser("normalize")
    normalize_parser.add_argument("--input", required=True)
    normalize_parser.add_argument("--output", required=True)
    normalize_parser.add_argument(
        "--split",
        required=True,
        choices=["smoke", "train", "dev", "test", "round2", "teacher"],
    )
    normalize_parser.set_defaults(func=_cmd_normalize)

    context_parser = subparsers.add_parser("context")
    context_parser.add_argument(
        "--output",
        default="tmp/data/processed/runtime_context.json",
    )
    context_parser.set_defaults(func=_cmd_context)

    dataset_parser = subparsers.add_parser("build-dataset")
    dataset_parser.add_argument("--input", required=True, nargs="+")
    dataset_parser.add_argument("--failure-source")
    dataset_parser.add_argument("--output-dir", required=True)
    dataset_parser.add_argument("--target-tokens", type=int, default=800_000)
    dataset_parser.add_argument("--seed", type=int, default=20261103)
    dataset_parser.set_defaults(func=_cmd_build_dataset)

    rl_parser = subparsers.add_parser("build-rl")
    rl_parser.add_argument(
        "--baseline-input",
        default="tmp/data/processed/train.jsonl",
    )
    rl_parser.add_argument(
        "--output-dir",
        default="tmp/data/processed/rl",
    )
    rl_parser.add_argument(
        "--task-split",
        choices=["train_pool", "dev"],
        default="train_pool",
    )
    rl_parser.add_argument("--seed", type=int, default=20261103)
    rl_parser.set_defaults(func=_cmd_build_rl)

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--input", required=True)
    summary_parser.add_argument("--output", required=True)
    summary_parser.set_defaults(func=_cmd_summarize)

    compare_parser = subparsers.add_parser("compare-tasks")
    compare_parser.add_argument("--left", required=True)
    compare_parser.add_argument("--right", required=True)
    compare_parser.add_argument("--output", required=True)
    compare_parser.add_argument("--bootstrap", type=int, default=10_000)
    compare_parser.add_argument("--seed", type=int, default=20261103)
    compare_parser.set_defaults(func=_cmd_compare_tasks)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
