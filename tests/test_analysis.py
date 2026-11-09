import pytest

from telecom_post_training.analysis import (
    paired_task_bootstrap,
    summarize,
    task_cluster_summary,
    wilson_interval,
)


def _record(task_id: str, success: bool, seed: int) -> dict:
    return {
        "task_id": task_id,
        "seed": seed,
        "strict_success": success,
        "termination_reason": "user_stop" if success else "max_steps",
        "failure_tags": ["success"] if success else ["max_steps"],
        "turn_count": 1,
        "tool_call_count": 0,
        "duration_seconds": 1,
        "user_cost": 0,
    }


def test_wilson_interval_is_bounded() -> None:
    low, high = wilson_interval(8, 10)
    assert 0 <= low < 0.8 < high <= 1


def test_summary_counts_failures() -> None:
    summary = summarize(
        [
            {
                "strict_success": True,
                "termination_reason": "user_stop",
                "failure_tags": ["success"],
                "turn_count": 8,
                "tool_call_count": 3,
                "duration_seconds": 10,
                "user_cost": 0.01,
            },
            {
                "strict_success": False,
                "termination_reason": "max_steps",
                "failure_tags": ["max_steps"],
                "turn_count": 10,
                "tool_call_count": 5,
                "duration_seconds": 20,
                "user_cost": 0.02,
            },
        ]
    )
    assert summary["success_rate"] == pytest.approx(0.5)
    assert summary["failures"] == {"max_steps": 1}
    assert summary["average_turns"] == 9


def test_task_cluster_summary_weights_tasks_equally() -> None:
    records = [
        _record("task-a", True, 1),
        _record("task-a", True, 2),
        _record("task-a", True, 3),
        _record("task-b", False, 1),
    ]
    summary = task_cluster_summary(records, n_bootstrap=100, seed=1)
    assert summary["task_clusters"] == 2
    assert summary["task_success_rate_mean"] == pytest.approx(0.5)
    assert summary["task_outcomes"] == {
        "all_fail": 1,
        "partial": 0,
        "all_success": 1,
    }
    assert summary["episodes_per_task_min"] == 1
    assert summary["episodes_per_task_max"] == 3


def test_paired_bootstrap_resamples_task_clusters_deterministically() -> None:
    left = []
    right = []
    left_counts = {"task-a": 3, "task-b": 2, "task-c": 3, "task-d": 1}
    right_counts = {"task-a": 2, "task-b": 1, "task-c": 3, "task-d": 1}
    for task_id, successes in left_counts.items():
        for index in range(3):
            left.append(_record(task_id, index < successes, index))
            right.append(_record(task_id, index < right_counts[task_id], index))

    first = paired_task_bootstrap(left, right, n_bootstrap=500, seed=19)
    second = paired_task_bootstrap(left, right, n_bootstrap=500, seed=19)
    assert first == second
    assert first["task_clusters"] == 4
    assert first["absolute_difference"] == pytest.approx(1 / 6)


def test_paired_bootstrap_requires_identical_task_clusters() -> None:
    with pytest.raises(ValueError, match="identical task clusters"):
        paired_task_bootstrap(
            [_record("task-a", True, 1)],
            [_record("task-b", True, 1)],
            n_bootstrap=10,
        )
