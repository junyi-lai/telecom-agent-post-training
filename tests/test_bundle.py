import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_bundle",
    ROOT / "scripts" / "build_bundle.py",
)
assert SPEC and SPEC.loader
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".venv/pyvenv.cfg",
        ".tools/bin/uv",
        "tmp/data/processed/train.jsonl",
        "checkpoints/sft/model.safetensors",
        "third_party/tau2-bench/file.py",
        ".git/modules/tau2/HEAD",
        "src/__pycache__/module.pyc",
        "keys/id_rsa",
        "keys/client.pem",
    ],
)
def test_bundle_rejects_intermediate_or_sensitive_paths(path: str) -> None:
    assert bundle._forbidden_path(Path(path))


def test_bundle_allows_required_training_files() -> None:
    allowed = (
        "models/Qwen3.5-2B/model.safetensors-00001-of-00001.safetensors",
        "results/data/processed/sft/sft_uniform.parquet",
        "results/reports/train_summary.md",
        "src/telecom_post_training/cli.py",
        ".git/HEAD",
    )
    assert all(not bundle._forbidden_path(Path(path)) for path in allowed)


def test_bundle_secret_scan_rejects_api_key(tmp_path: Path) -> None:
    source = tmp_path / "source"
    stage = tmp_path / "stage"
    source.mkdir()
    stage.mkdir()
    (source / ".env").write_text(
        "DEEPSEEK_API_KEY=secret-value\n",
        encoding="utf-8",
    )
    (stage / "config.txt").write_text(
        "token=secret-value\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="API key"):
        bundle._secret_scan(source, stage)


def test_bundle_secret_scan_rejects_private_key(tmp_path: Path) -> None:
    source = tmp_path / "source"
    stage = tmp_path / "stage"
    source.mkdir()
    stage.mkdir()
    (stage / "notes.txt").write_text(
        f"{'-' * 5}BEGIN OPENSSH PRIVATE KEY{'-' * 5}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Private key"):
        bundle._secret_scan(source, stage)
