from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_generated_outputs_use_tmp_and_results_layout() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    cli = (ROOT / "src" / "telecom_post_training" / "cli.py").read_text(
        encoding="utf-8"
    )
    collect = (ROOT / "src" / "telecom_post_training" / "collect.py").read_text(
        encoding="utf-8"
    )
    teacher = (ROOT / "src" / "telecom_post_training" / "teacher.py").read_text(
        encoding="utf-8"
    )
    assert "TMP ?= tmp" in makefile
    assert 'default="tmp/data/processed/runtime_context.json"' in cli
    assert '"tmp" / "data" / "processed" / "runtime_context.json"' in cli
    assert 'ROOT / "tmp" / "artifacts" / "runs" / args.stage' in collect
    assert 'ROOT / "tmp" / "artifacts" / "runs" / "teacher"' in teacher


def test_results_keep_only_curated_outputs() -> None:
    required = (
        "results/README.md",
        "results/reports/train_summary.md",
        "results/reports/teacher_summary.md",
        "results/reports/sft_selection_audit.md",
        "results/reports/rl_prompt_audit.md",
    )
    assert all((ROOT / path).exists() for path in required)
    assert not (ROOT / "artifacts").exists()
    assert not (ROOT / "data" / "processed").exists()


def test_tmp_and_cache_configuration() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    gitattributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "tmp/*" in gitignore
    assert "!/results/reports/**" in gitignore
    assert "!/results/data/**" in gitignore
    assert "!/results/runs/baseline/results.json" in gitignore
    assert "/results/runs/baseline/collection_metadata.json" in gitignore
    assert "/results/runs/*" in gitignore
    assert "* text=auto eol=lf" in gitattributes
    assert "*.parquet binary" in gitattributes
    assert "*.safetensors binary" in gitattributes
    assert 'cache_dir = "tmp/pytest_cache"' in pyproject
    assert 'cache-dir = "tmp/ruff_cache"' in pyproject


def test_publishable_training_data_is_present() -> None:
    required = (
        "data/SHA256SUMS",
        "results/data/processed/train.jsonl",
        "results/data/processed/teacher.jsonl",
        "results/data/processed/sft/sft_uniform.parquet",
        "results/data/processed/sft/sft_failure_aware.parquet",
        "results/data/processed/rl/rl_uniform.parquet",
        "results/data/processed/rl/rl_failure_aware.parquet",
        "results/data/processed/rl_dev/rl_uniform.parquet",
        "results/data/processed/rl_dev/rl_failure_aware.parquet",
        "results/runs/baseline/results.json",
        "results/runs/teacher/results.json",
    )
    assert all((ROOT / path).exists() for path in required)


def test_no_intermediate_directories_outside_dependency_trees() -> None:
    blocked = {
        ".pytest_cache",
        ".ruff_cache",
        "artifacts",
        "checkpoints",
    }
    assert not any((ROOT / name).exists() for name in blocked)
    assert not (ROOT / "data" / "processed").exists()
    assert not (ROOT / "data" / "eval").exists()
