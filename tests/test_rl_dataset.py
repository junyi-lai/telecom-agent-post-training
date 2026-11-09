from pathlib import Path

from telecom_post_training.manifest import build_manifest
from telecom_post_training.rl_dataset import build_rl_prompts


def test_failure_aware_rl_prompts_cover_every_task_before_weighted_extras(
    tmp_path: Path,
) -> None:
    manifest = build_manifest(Path("third_party/tau2-bench"))
    context = {"system_prompt": "system", "tools": []}
    selected = manifest["splits"]["train_pool"][0]
    records = [
        {
            "task_id": selected,
            "strict_success": False,
            "failure_tags": ["wrong_action_or_policy"],
        }
    ]
    summary = build_rl_prompts(
        manifest,
        context,
        records,
        output_dir=tmp_path,
        seed=1,
    )
    assert summary["uniform_rows"] == 164
    assert summary["failure_aware_rows"] == 328
    assert summary["uniform_unique_tasks"] == 164
    assert summary["failure_aware_unique_tasks"] == 164
    assert summary["uniform_task_occurrence_distribution"] == {1: 164}
    assert summary["failure_aware_extra_rows"] == 164
    assert set(summary["failure_aware_task_occurrences"]) == set(
        manifest["splits"]["train_pool"]
    )
    assert min(summary["failure_aware_task_occurrences"].values()) >= 1
    assert summary["task_split"] == "train_pool"
    repeated = build_rl_prompts(
        manifest,
        context,
        records,
        output_dir=tmp_path / "repeat",
        seed=1,
    )
    assert (
        repeated["failure_aware_task_occurrences"]
        == summary["failure_aware_task_occurrences"]
    )


def test_dev_prompts_match_manifest_dev_split(tmp_path: Path) -> None:
    manifest = build_manifest(Path("third_party/tau2-bench"))
    context = {"system_prompt": "system", "tools": []}
    summary = build_rl_prompts(
        manifest,
        context,
        [],
        output_dir=tmp_path,
        seed=1,
        task_split="dev",
    )
    assert summary["uniform_rows"] == 10
    assert summary["failure_aware_rows"] == 20
    assert summary["uniform_unique_tasks"] == 10
    assert summary["failure_aware_unique_tasks"] == 10
    assert summary["uniform_task_occurrence_distribution"] == {1: 10}
    assert summary["failure_aware_extra_rows"] == 10
