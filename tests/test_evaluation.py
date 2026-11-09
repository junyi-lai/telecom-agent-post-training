from __future__ import annotations

import json
from pathlib import Path

from telecom_post_training.evaluation import validate_evaluation


def _write_results(path: Path, simulations: list[dict]) -> Path:
    path.write_text(
        json.dumps({"simulations": simulations}),
        encoding="utf-8",
    )
    return path


def test_validate_evaluation_accepts_all_task_outcomes(tmp_path: Path) -> None:
    results = _write_results(
        tmp_path / "results.json",
        [
            {"task_id": "task-a", "termination_reason": "user_stop"},
            {"task_id": "task-b", "termination_reason": "max_steps"},
        ],
    )

    validation = validate_evaluation(
        results,
        expected_task_ids=["task-a", "task-b"],
    )

    assert validation.valid
    assert validation.infrastructure_errors == 0
    assert validation.termination_counts == {
        "max_steps": 1,
        "user_stop": 1,
    }


def test_validate_evaluation_rejects_infrastructure_failures(
    tmp_path: Path,
) -> None:
    results = _write_results(
        tmp_path / "results.json",
        [
            {"task_id": "task-a", "termination_reason": "user_stop"},
            {
                "task_id": "task-b",
                "termination_reason": "infrastructure_error",
                "info": {"error_type": "DeepSeekTotalDeadlineExceeded"},
            },
        ],
    )

    validation = validate_evaluation(
        results,
        expected_task_ids=["task-a", "task-b"],
    )

    assert not validation.valid
    assert validation.infrastructure_errors == 1
    assert validation.errors == ["infrastructure errors: 1"]


def test_validate_evaluation_detects_missing_and_duplicate_tasks(
    tmp_path: Path,
) -> None:
    results = _write_results(
        tmp_path / "results.json",
        [
            {"task_id": "task-a", "termination_reason": "user_stop"},
            {"task_id": "task-a", "termination_reason": "max_steps"},
            {"task_id": "task-c", "termination_reason": "user_stop"},
        ],
    )

    validation = validate_evaluation(
        results,
        expected_task_ids=["task-a", "task-b", "task-c"],
    )

    assert not validation.valid
    assert validation.missing_tasks == ["task-b"]
    assert validation.unexpected_tasks == []
    assert validation.duplicate_tasks == ["task-a"]
