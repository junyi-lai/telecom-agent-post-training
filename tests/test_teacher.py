from pathlib import Path

from telecom_post_training.settings import ROOT, Settings
from telecom_post_training.teacher import build_command


def test_teacher_command_uses_ground_truth_agent(tmp_path: Path) -> None:
    settings = Settings(
        student_model="Qwen/Qwen3.5-2B",
        student_model_revision="revision",
        student_model_path=tmp_path / "missing-model",
        vllm_model_alias="student",
        deepseek_base_url="https://api.deepseek.com/v1",
        deepseek_api_key="secret-value",
        deepseek_model="deepseek-chat",
        deepseek_teacher_model="deepseek-v4-pro",
        tau2_repo=ROOT / "third_party" / "tau2-bench",
    )
    command = build_command(
        settings,
        task_ids=["task-a"],
        trials=2,
        output=tmp_path / "teacher",
    )
    assert "llm_agent_gt" in command
    assert "deepseek/deepseek-chat" in command
    assert "secret-value" not in " ".join(command)
