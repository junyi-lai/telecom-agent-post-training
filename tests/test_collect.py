from pathlib import Path

from telecom_post_training.collect import build_command
from telecom_post_training.serve import build_command as build_serve_command
from telecom_post_training.settings import ROOT, Settings


def _settings(*, model_path: Path) -> Settings:
    return Settings(
        student_model="Qwen/Qwen3.5-2B",
        student_model_revision="revision",
        student_model_path=model_path,
        vllm_model_alias="student",
        deepseek_base_url="https://api.deepseek.com/v1",
        deepseek_api_key="secret-value",
        deepseek_model="deepseek-chat",
        deepseek_teacher_model="deepseek-v4-pro",
        tau2_repo=ROOT / "third_party" / "tau2-bench",
    )


def test_collect_command_does_not_embed_api_key(tmp_path: Path) -> None:
    command = build_command(
        _settings(model_path=tmp_path / "missing-model"),
        stage="smoke",
        trials=2,
        max_concurrency=2,
        max_steps=30,
        seed=20261103,
        output=tmp_path / "run",
    )
    command_text = " ".join(command)
    assert "secret-value" not in command_text
    assert "deepseek/deepseek-chat" in command
    assert "openai/student" in command


def test_serve_command_uses_local_model_and_language_only(tmp_path: Path) -> None:
    command = build_serve_command(_settings(model_path=tmp_path))
    assert str(tmp_path) in command
    assert "--language-model-only" in command
    assert "qwen3_coder" in command
