import json
from collections import Counter

import pyarrow.parquet as pq
import pytest

from telecom_post_training import dataset as dataset_module
from telecom_post_training.dataset import (
    build_conversation_examples,
    build_datasets,
    coverage_example_fingerprints,
    select_budget,
    write_parquet,
)


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(max(1, len(text.split()))))


def _example(
    index: int,
    *,
    task_id: str,
    failure_bucket: str = "success",
    assistant_tokens: int = 10,
) -> dict:
    return {
        "id": f"sim-{index}",
        "task_id": task_id,
        "failure_bucket": failure_bucket,
        "assistant_token_count": assistant_tokens,
        "messages": [
            {"role": "user", "content": f"task={task_id}, trial={index}"},
            {"role": "assistant", "content": f"response-{index}"},
        ],
    }


def test_conversation_examples_include_tools_and_full_history() -> None:
    context = {"system_prompt": "system", "tools": [{"type": "function"}]}
    records = [
        {
            "simulation_id": "sim",
            "task_id": "task",
            "split": "train",
            "strict_success": True,
            "failure_tags": ["success"],
            "messages": [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "name": "check_status",
                            "arguments": {},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "ok",
                    "tool_call_id": "call-1",
                },
            ],
        }
    ]
    examples = build_conversation_examples(
        records,
        context=context,
        tokenizer=FakeTokenizer(),
    )
    assert len(examples) == 1
    assert examples[0]["messages"][0]["role"] == "system"
    assert examples[0]["messages"][2]["tool_calls"][0]["function"]["name"] == (
        "check_status"
    )


def test_failed_task_with_max_steps_is_not_bucketed_as_success() -> None:
    context = {"system_prompt": "system", "tools": []}
    records = [
        {
            "simulation_id": "sim",
            "task_id": "task",
            "split": "teacher",
            "strict_success": True,
            "failure_tags": ["success"],
            "source_failure_tags": ["max_steps"],
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        }
    ]
    examples = build_conversation_examples(
        records,
        context=context,
        tokenizer=FakeTokenizer(),
    )
    assert examples[0]["failure_bucket"] == "max_steps"


def test_identical_message_trajectories_are_deduplicated() -> None:
    context = {"system_prompt": "system", "tools": []}
    shared_messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    records = [
        {
            "simulation_id": simulation_id,
            "task_id": "task",
            "split": "train",
            "strict_success": True,
            "failure_tags": ["success"],
            "messages": shared_messages,
        }
        for simulation_id in ("sim-a", "sim-b")
    ]
    examples = build_conversation_examples(
        records,
        context=context,
        tokenizer=FakeTokenizer(),
    )
    assert len(examples) == 1


@pytest.mark.parametrize("strategy", ["uniform", "failure_aware"])
def test_both_strategies_apply_per_task_cap(strategy: str) -> None:
    examples = [
        _example(index, task_id="task-a", failure_bucket="success")
        for index in range(4)
    ]
    examples += [
        _example(index + 4, task_id="task-b", failure_bucket="repetition")
        for index in range(4)
    ]
    selected = select_budget(
        examples,
        strategy=strategy,
        target_tokens=10_000,
        seed=7,
    )
    counts = Counter(example["task_id"] for example in selected)
    assert counts == {"task-a": 2, "task-b": 2}


@pytest.mark.parametrize("strategy", ["uniform", "failure_aware"])
def test_coverage_pass_includes_every_required_task_before_seconds(
    strategy: str,
) -> None:
    examples = [
        _example(0, task_id="task-a", assistant_tokens=5),
        _example(1, task_id="task-a", assistant_tokens=100),
        _example(2, task_id="task-b", assistant_tokens=7),
        _example(3, task_id="task-b", assistant_tokens=100),
        _example(4, task_id="task-c", assistant_tokens=9),
    ]
    selected = select_budget(
        examples,
        strategy=strategy,
        target_tokens=21,
        seed=11,
        required_task_ids=["task-a", "task-b", "task-c"],
    )
    assert {example["task_id"] for example in selected} == {
        "task-a",
        "task-b",
        "task-c",
    }
    assert Counter(example["task_id"] for example in selected) == {
        "task-a": 1,
        "task-b": 1,
        "task-c": 1,
    }
    assert {example["id"] for example in selected} == {
        "sim-0",
        "sim-2",
        "sim-4",
    }


def test_coverage_fails_when_required_task_has_no_successful_trajectory() -> None:
    with pytest.raises(ValueError, match="have no successful trajectory"):
        select_budget(
            [_example(0, task_id="task-a")],
            strategy="uniform",
            target_tokens=1_000,
            seed=1,
            required_task_ids=["task-a", "task-b"],
        )


def test_build_datasets_exports_every_required_task(
    tmp_path, monkeypatch
) -> None:
    records = []
    for task_index in range(3):
        for trial in range(2):
            records.append(
                {
                    "simulation_id": f"sim-{task_index}-{trial}",
                    "task_id": f"task-{task_index}",
                    "split": "train",
                    "strict_success": True,
                    "failure_tags": ["success"],
                    "messages": [
                        {"role": "user", "content": f"task {task_index}"},
                        {
                            "role": "assistant",
                            "content": f"answer {task_index} {trial}",
                        },
                    ],
                }
            )
    source = tmp_path / "records.jsonl"
    source.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        dataset_module,
        "build_runtime_context",
        lambda: {"system_prompt": "system", "tools": [], "sha256": "test"},
    )
    summary = build_datasets(
        [source],
        tmp_path / "sft",
        tokenizer=FakeTokenizer(),
        target_tokens=1_000,
        seed=3,
        required_task_ids=["task-0", "task-1", "task-2"],
    )
    assert summary["required_task_count"] == 3
    for arm in ("uniform", "failure_aware"):
        assert summary["arms"][arm]["unique_tasks"] == 3
        assert summary["arms"][arm]["uncovered_tasks"] == []
    assert summary["cross_arm"]["shared_trajectories"] == 3
    assert summary["cross_arm"]["uniform_only_trajectories"] == 3
    assert summary["cross_arm"]["failure_aware_only_trajectories"] == 0


def test_failure_bucket_rotation_also_has_per_task_cap() -> None:
    examples = [
        _example(index, task_id="task-a", failure_bucket="invalid_tool_call")
        for index in range(4)
    ]
    examples += [
        _example(index + 4, task_id="task-b", failure_bucket="max_steps")
        for index in range(4)
    ]
    selected = select_budget(
        examples,
        strategy="failure_aware",
        target_tokens=10_000,
        seed=3,
    )
    counts = Counter(example["task_id"] for example in selected)
    assert counts == {"task-a": 2, "task-b": 2}
    assert len(selected) == len({example["id"] for example in selected})


def test_failure_aware_second_pass_excludes_uniform_seconds() -> None:
    examples = [
        _example(
            task_index * 4 + trial,
            task_id=f"task-{task_index}",
            failure_bucket=(
                "invalid_tool_call" if task_index % 2 else "max_steps"
            ),
        )
        for task_index in range(3)
        for trial in range(4)
    ]
    required = ["task-0", "task-1", "task-2"]
    coverage = coverage_example_fingerprints(
        examples,
        required_task_ids=required,
        seed=5,
    )
    uniform = select_budget(
        examples,
        strategy="uniform",
        target_tokens=10_000,
        seed=5,
        required_task_ids=required,
    )
    uniform_fingerprints = {
        dataset_module._example_fingerprint(example) for example in uniform
    }
    uniform_seconds = uniform_fingerprints - coverage
    failure_aware = select_budget(
        examples,
        strategy="failure_aware",
        target_tokens=10_000,
        seed=5,
        required_task_ids=required,
        excluded_second_pass_fingerprints=uniform_seconds,
    )
    failure_fingerprints = {
        dataset_module._example_fingerprint(example) for example in failure_aware
    }
    assert coverage <= failure_fingerprints
    assert not (uniform_seconds & (failure_fingerprints - coverage))
    assert {example["task_id"] for example in failure_aware} == set(required)


@pytest.mark.parametrize("strategy", ["uniform", "failure_aware"])
def test_budget_selection_is_deterministic(strategy: str) -> None:
    examples = [
        _example(
            index,
            task_id=f"task-{index}",
            failure_bucket="success" if index < 5 else "repetition",
        )
        for index in range(10)
    ]
    first = select_budget(
        examples,
        strategy=strategy,
        target_tokens=120,
        seed=1,
    )
    second = select_budget(
        examples,
        strategy=strategy,
        target_tokens=120,
        seed=1,
    )
    assert [item["id"] for item in first] == [item["id"] for item in second]
    assert sum(item["assistant_token_count"] for item in first) <= 120


def test_parquet_export_is_readable(tmp_path) -> None:
    rows = [
        {
            "id": "sim",
            "task_id": "task",
            "split": "train",
            "failure_bucket": "success",
            "source_failure_tags": ["success"],
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi", "tool_calls": None},
            ],
            "tools": [],
            "enable_thinking": False,
            "assistant_token_count": 1,
        }
    ]
    destination = tmp_path / "sft.parquet"
    write_parquet(rows, destination)
    table = pq.read_table(destination)
    assert table.num_rows == 1
    assert table.column("id").to_pylist() == ["sim"]
