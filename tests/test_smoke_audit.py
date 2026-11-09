import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_grpo_smoke_audit_accepts_healthy_group_variation(tmp_path: Path) -> None:
    context_log = tmp_path / "context.jsonl"
    episodes = []
    for task_index in range(8):
        for group_index in range(8):
            episodes.append(
                {
                    "task_id": f"task-{task_index}",
                    "reward_score": float(group_index % 2),
                    "prompt_token_lengths": [100, 110, 120],
                    "tool_calls": 1,
                    "invalid_tool_calls": 0,
                }
            )
    context_log.write_text(
        "\n".join(json.dumps(episode) for episode in episodes) + "\n",
        encoding="utf-8",
    )
    deepseek_log = tmp_path / "deepseek.jsonl"
    deepseek_log.write_text(
        json.dumps(
            {
                "event": "retry",
                "status_code": None,
                "timeout": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    train_log = tmp_path / "train.log"
    train_log.write_text("loss=0.4 reward=0.2\n", encoding="utf-8")
    output = tmp_path / "audit.json"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_grpo_smoke.py"),
            "--context-log",
            str(context_log),
            "--deepseek-log",
            str(deepseek_log),
            "--train-log",
            str(train_log),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["agent_episodes"] == 64
    assert result["variable_reward_groups"] == 8
    assert result["valid_tool_call_episodes"] == 64
    assert result["deepseek_429_events"] == 0


def test_grpo_smoke_audit_ignores_partial_refill_group(tmp_path: Path) -> None:
    context_log = tmp_path / "context.jsonl"
    episodes = []
    for task_index in range(8):
        for group_index in range(8):
            episodes.append(
                {
                    "task_id": f"task-{task_index}",
                    "reward_score": float(group_index % 2),
                    "prompt_token_lengths": [100, 110],
                    "tool_calls": 1,
                    "invalid_tool_calls": 0,
                }
            )
    episodes.append(
        {
            "task_id": "partial-refill-task",
            "reward_score": 0.0,
            "prompt_token_lengths": [100, 105],
            "tool_calls": 1,
            "invalid_tool_calls": 0,
        }
    )
    context_log.write_text(
        "\n".join(json.dumps(episode) for episode in episodes) + "\n",
        encoding="utf-8",
    )
    deepseek_log = tmp_path / "deepseek.jsonl"
    deepseek_log.write_text("", encoding="utf-8")
    train_log = tmp_path / "train.log"
    train_log.write_text("loss=0.4 reward=0.2\n", encoding="utf-8")
    output = tmp_path / "audit.json"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_grpo_smoke.py"),
            "--context-log",
            str(context_log),
            "--deepseek-log",
            str(deepseek_log),
            "--train-log",
            str(train_log),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["task_groups"] == 8
    assert result["partial_or_evicted_task_groups"] == 1


def test_grpo_smoke_audit_records_bounded_rate_limit_retries(
    tmp_path: Path,
) -> None:
    context_log = tmp_path / "context.jsonl"
    episodes = [
        {
            "task_id": f"task-{task_index // 8}",
            "reward_score": float(task_index % 2),
            "prompt_token_lengths": [100, 110],
            "tool_calls": 1,
            "invalid_tool_calls": 0,
        }
        for task_index in range(64)
    ]
    context_log.write_text(
        "\n".join(json.dumps(episode) for episode in episodes) + "\n",
        encoding="utf-8",
    )
    deepseek_log = tmp_path / "deepseek.jsonl"
    deepseek_log.write_text(
        "\n".join(
            json.dumps(
                {
                    "event": "retry",
                    "status_code": 429,
                    "timeout": False,
                }
            )
            for _ in range(12)
        )
        + "\n",
        encoding="utf-8",
    )
    train_log = tmp_path / "train.log"
    train_log.write_text("loss=0.4 reward=0.2\n", encoding="utf-8")
    output = tmp_path / "audit.json"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_grpo_smoke.py"),
            "--context-log",
            str(context_log),
            "--deepseek-log",
            str(deepseek_log),
            "--train-log",
            str(train_log),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["deepseek_429_events"] == 12
    assert result["deepseek_retry_events"] == 12
    assert result["deepseek_failure_events"] == 0


def test_grpo_smoke_audit_retry_budget_scales_to_one_per_episode(
    tmp_path: Path,
) -> None:
    context_log = tmp_path / "context.jsonl"
    episodes = [
        {
            "task_id": f"task-{task_index // 8}",
            "reward_score": float(task_index % 2),
            "prompt_token_lengths": [100, 110],
            "tool_calls": 1,
            "invalid_tool_calls": 0,
        }
        for task_index in range(64)
    ]
    context_log.write_text(
        "\n".join(json.dumps(episode) for episode in episodes) + "\n",
        encoding="utf-8",
    )
    deepseek_log = tmp_path / "deepseek.jsonl"
    train_log = tmp_path / "train.log"
    train_log.write_text("loss=0.4 reward=0.2\n", encoding="utf-8")
    output = tmp_path / "audit.json"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "audit_grpo_smoke.py"),
        "--context-log",
        str(context_log),
        "--deepseek-log",
        str(deepseek_log),
        "--train-log",
        str(train_log),
        "--output",
        str(output),
    ]
    retry = {
        "event": "retry",
        "status_code": 500,
        "timeout": False,
    }
    deepseek_log.write_text(
        "\n".join(json.dumps(retry) for _ in range(64)) + "\n",
        encoding="utf-8",
    )
    subprocess.run(command, check=True, capture_output=True, text=True)

    deepseek_log.write_text(
        "\n".join(json.dumps(retry) for _ in range(65)) + "\n",
        encoding="utf-8",
    )
    exceeded = subprocess.run(command, capture_output=True, text=True)
    assert exceeded.returncode != 0
    assert "DeepSeek retry events: 65" in exceeded.stderr
