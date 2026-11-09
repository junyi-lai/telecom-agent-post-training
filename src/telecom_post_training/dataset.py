from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from telecom_post_training.context import build_runtime_context

FAILURE_WEIGHTS = {
    "success": 40,
    "invalid_tool_call": 12,
    "repetition": 12,
    "wrong_action_or_policy": 12,
    "incomplete_state": 12,
    "premature_exit": 12,
    "max_steps": 12,
    "response_truncated": 6,
    "agent_error": 6,
    "generic_failure": 6,
}

FAILURE_PRIORITY = (
    "wrong_action_or_policy",
    "invalid_tool_call",
    "incomplete_state",
    "repetition",
    "premature_exit",
    "max_steps",
    "response_truncated",
    "agent_error",
    "generic_failure",
)
MAX_EXAMPLES_PER_TASK = 2


def _rank(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _message_fingerprint(messages: list[dict]) -> str:
    payload = json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _assistant_to_openai(message: dict) -> dict:
    calls = []
    for call in message.get("tool_calls") or []:
        calls.append(
            {
                "id": call.get("id"),
                "type": "function",
                "function": {
                    "name": call.get("name"),
                    "arguments": json.dumps(
                        call.get("arguments") or {},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            }
        )
    return {
        "role": "assistant",
        "content": message.get("content"),
        "tool_calls": calls or None,
    }


def _history_message(message: dict) -> dict:
    role = message.get("role")
    if role == "assistant":
        return _assistant_to_openai(message)
    if role == "tool":
        return {
            "role": "tool",
            "content": message.get("content"),
            "tool_call_id": message.get("tool_call_id"),
        }
    if role == "system":
        return {"role": "system", "content": message.get("content")}
    # tau2's user tool calls are delivered to the Agent as separate tool messages.
    return {"role": "user", "content": message.get("content")}


def _target_token_count(tokenizer, message: dict) -> int:
    content = message.get("content") or ""
    serialized_calls = json.dumps(
        message.get("tool_calls") or [],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return len(
        tokenizer.encode(
            content + "\n" + serialized_calls,
            add_special_tokens=False,
        )
    )


def _conversation_token_count(
    tokenizer, messages: list[dict], tools: list[dict]
) -> int:
    try:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
        )
        return len(token_ids)
    except (AttributeError, TypeError, ValueError):
        return sum(
            len(
                tokenizer.encode(
                    json.dumps(message, ensure_ascii=False),
                    add_special_tokens=False,
                )
            )
            for message in messages
        )


def _primary_bucket(record: dict) -> str:
    tags = record.get("source_failure_tags") or record.get("failure_tags") or []
    for bucket in FAILURE_PRIORITY:
        if bucket in tags:
            return bucket
    if record.get("strict_success", False):
        return "success"
    return "generic_failure"


def _trajectory_messages(record: dict, system_prompt: str) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(
        _history_message(message) for message in record.get("messages") or []
    )
    return messages


def build_conversation_examples(
    records: list[dict],
    *,
    context: dict,
    tokenizer,
) -> list[dict]:
    examples: list[dict] = []
    seen_message_fingerprints: set[str] = set()
    for record in records:
        if not record.get("strict_success"):
            continue
        messages = record.get("messages") or []
        assistant_token_count = sum(
            _target_token_count(tokenizer, _assistant_to_openai(message))
            for message in messages
            if message.get("role") == "assistant"
        )
        chat_messages = _trajectory_messages(record, context["system_prompt"])
        total_token_count = _conversation_token_count(
            tokenizer,
            chat_messages,
            context["tools"],
        )
        if total_token_count > 32768:
            continue
        message_fingerprint = _message_fingerprint(chat_messages)
        if message_fingerprint in seen_message_fingerprints:
            continue
        seen_message_fingerprints.add(message_fingerprint)
        examples.append(
            {
                "id": record["simulation_id"],
                "task_id": record.get("task_id"),
                "split": record.get("split"),
                "failure_bucket": _primary_bucket(record),
                "source_failure_tags": record.get("failure_tags") or [],
                "messages": chat_messages,
                "tools": context["tools"],
                "enable_thinking": False,
                "assistant_token_count": assistant_token_count,
                "total_token_count": total_token_count,
                "message_fingerprint": message_fingerprint,
            }
        )
    return examples


def _task_id(example: dict) -> str:
    return str(example.get("task_id") or example["id"])


def _example_fingerprint(example: dict) -> str:
    if example.get("message_fingerprint"):
        return str(example["message_fingerprint"])
    return _message_fingerprint(example.get("messages") or [])


def _required_tasks(
    examples: list[dict],
    required_task_ids: Iterable[str] | None,
) -> list[str]:
    return (
        sorted(set(required_task_ids))
        if required_task_ids is not None
        else sorted({_task_id(example) for example in examples})
    )


def _coverage_examples(
    examples: list[dict],
    required_tasks: list[str],
    seed: int,
) -> list[dict]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for example in examples:
        by_task[_task_id(example)].append(example)
    missing_tasks = [task_id for task_id in required_tasks if task_id not in by_task]
    if missing_tasks:
        raise ValueError(
            f"{len(missing_tasks)} required tasks have no successful trajectory: "
            f"{missing_tasks}"
        )
    first_choices = [
        min(
            by_task[task_id],
            key=lambda example: (
                int(example["assistant_token_count"]),
                _rank(example["id"], seed),
            ),
        )
        for task_id in required_tasks
    ]
    first_choices.sort(key=lambda example: _rank(_task_id(example), seed))
    return first_choices


def coverage_example_fingerprints(
    examples: list[dict],
    *,
    required_task_ids: Iterable[str] | None,
    seed: int,
) -> set[str]:
    required_tasks = _required_tasks(examples, required_task_ids)
    return {
        _example_fingerprint(example)
        for example in _coverage_examples(examples, required_tasks, seed)
    }


def select_budget(
    examples: list[dict],
    *,
    strategy: str,
    target_tokens: int,
    seed: int,
    required_task_ids: Iterable[str] | None = None,
    excluded_second_pass_fingerprints: Iterable[str] = (),
) -> list[dict]:
    if strategy not in {"uniform", "failure_aware"}:
        raise ValueError(f"Unknown strategy: {strategy}")

    required_tasks = _required_tasks(examples, required_task_ids)
    first_choices = _coverage_examples(examples, required_tasks, seed)
    coverage_tokens = sum(
        int(example["assistant_token_count"]) for example in first_choices
    )
    if coverage_tokens > target_tokens:
        raise ValueError(
            "The assistant-token budget cannot cover every required task: "
            f"minimum={coverage_tokens}, target={target_tokens}"
        )

    selected: list[dict] = []
    selected_ids: set[str] = set()
    selected_fingerprints: set[str] = set()
    selected_task_counts: Counter[str] = Counter()
    token_count = 0

    def add_if_available(example: dict, remaining: int) -> bool:
        example_id = str(example.get("id"))
        fingerprint = _example_fingerprint(example)
        task_id = _task_id(example)
        cost = max(1, int(example["assistant_token_count"]))
        if example_id in selected_ids or fingerprint in selected_fingerprints:
            return False
        if selected_task_counts[task_id] >= MAX_EXAMPLES_PER_TASK:
            return False
        if cost > remaining:
            return False
        selected.append(example)
        selected_ids.add(example_id)
        selected_fingerprints.add(fingerprint)
        selected_task_counts[task_id] += 1
        return True

    for example in first_choices:
        remaining = target_tokens - token_count
        if not add_if_available(example, remaining):
            raise ValueError("Coverage selection unexpectedly exceeded the budget")
        token_count += int(example["assistant_token_count"])

    excluded_second_pass = set(excluded_second_pass_fingerprints)
    remaining_examples = [
        example
        for example in examples
        if str(example.get("id")) not in selected_ids
        and _task_id(example) in required_tasks
        and _example_fingerprint(example) not in excluded_second_pass
    ]
    if strategy == "uniform":
        second_pass = sorted(
            remaining_examples,
            key=lambda example: _rank(example["id"], seed),
        )
    else:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for example in remaining_examples:
            grouped[example["failure_bucket"]].append(example)
        for values in grouped.values():
            values.sort(key=lambda example: _rank(example["id"], seed))

        second_pass = []
        offsets = {name: 0 for name in FAILURE_WEIGHTS}
        completed: set[str] = set()
        while len(completed) < len(FAILURE_WEIGHTS):
            for bucket, weight in FAILURE_WEIGHTS.items():
                values = grouped.get(bucket, [])
                for _ in range(weight):
                    offset = offsets.get(bucket, 0)
                    if offset < len(values):
                        second_pass.append(values[offset])
                        offsets[bucket] = offset + 1
                    else:
                        completed.add(bucket)
                        break
        ordered_ids = {example["id"] for example in second_pass}
        second_pass.extend(
            sorted(
                (
                    example
                    for example in remaining_examples
                    if example["id"] not in ordered_ids
                ),
                key=lambda example: _rank(example["id"], seed),
            )
        )

    for example in second_pass:
        remaining = target_tokens - token_count
        if add_if_available(example, remaining):
            token_count += int(example["assistant_token_count"])
        if target_tokens - token_count < 16:
            break

    covered_tasks = {_task_id(example) for example in selected}
    missing_after_selection = [
        task_id for task_id in required_tasks if task_id not in covered_tasks
    ]
    if missing_after_selection:
        raise ValueError(
            f"SFT selection left required tasks uncovered: {missing_after_selection}"
        )
    return selected


def _selection_summary(
    examples: list[dict],
    required_task_ids: Iterable[str] | None = None,
) -> dict:
    required_tasks = (
        sorted(set(required_task_ids))
        if required_task_ids is not None
        else sorted({_task_id(example) for example in examples})
    )
    task_counts = Counter(_task_id(example) for example in examples)
    fingerprinted = Counter(_example_fingerprint(example) for example in examples)
    duplicate_messages = sum(
        count - 1 for count in fingerprinted.values() if count > 1
    )
    duplicate_ids = sum(
        count - 1 for count in Counter(example.get("id") for example in examples).values()
        if count > 1
    )
    distribution = {
        str(count): sum(value == count for value in task_counts.values())
        for count in sorted(set(task_counts.values()))
    }
    uncovered_tasks = [
        task_id for task_id in required_tasks if task_id not in task_counts
    ]
    if task_counts and max(task_counts.values()) > MAX_EXAMPLES_PER_TASK:
        raise ValueError("Per-task cap was exceeded during SFT selection")
    if duplicate_messages or duplicate_ids:
        raise ValueError("SFT selection contains duplicate trajectories")
    if uncovered_tasks:
        raise ValueError(f"SFT selection is missing required tasks: {uncovered_tasks}")
    return {
        "examples": len(examples),
        "assistant_tokens": sum(
            int(example["assistant_token_count"]) for example in examples
        ),
        "unique_tasks": len(task_counts),
        "uncovered_task_count": len(uncovered_tasks),
        "uncovered_tasks": uncovered_tasks,
        "task_count_distribution": distribution,
        "max_examples_per_task": (
            max(task_counts.values()) if task_counts else 0
        ),
        "duplicate_message_trajectory_rows": duplicate_messages,
        "duplicate_example_id_rows": duplicate_ids,
    }


def write_examples(
    examples: list[dict],
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for example in examples:
            stream.write(json.dumps(example, ensure_ascii=False) + "\n")


def write_parquet(examples: list[dict], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if examples:
        table = pa.Table.from_pylist(examples)
    else:
        table = pa.table(
            {
                "id": pa.array([], type=pa.string()),
                "task_id": pa.array([], type=pa.string()),
                "split": pa.array([], type=pa.string()),
                "failure_bucket": pa.array([], type=pa.string()),
                "source_failure_tags": pa.array([], type=pa.list_(pa.string())),
                "messages": pa.array([], type=pa.list_(pa.string())),
                "tools": pa.array([], type=pa.list_(pa.string())),
                "enable_thinking": pa.array([], type=pa.bool_()),
                "assistant_token_count": pa.array([], type=pa.int64()),
                "total_token_count": pa.array([], type=pa.int64()),
                "message_fingerprint": pa.array([], type=pa.string()),
            }
        )
    pq.write_table(table, destination)


def build_datasets(
    record_paths: list[Path],
    output_dir: Path,
    *,
    tokenizer,
    target_tokens: int,
    seed: int,
    failure_source: Path | None = None,
    required_task_ids: Iterable[str] | None = None,
) -> dict:
    records = [record for path in record_paths for record in _load_jsonl(path)]
    failure_by_task: dict[str, list[str]] = {}
    if failure_source and failure_source.exists():
        for record in _load_jsonl(failure_source):
            task_id = record.get("task_id")
            tags = [
                tag
                for tag in record.get("failure_tags") or []
                if tag in FAILURE_WEIGHTS and tag != "success"
            ]
            if task_id and tags:
                failure_by_task.setdefault(task_id, []).extend(tags)
    for record in records:
        if record.get("strict_success") and record.get("task_id") in failure_by_task:
            record["source_failure_tags"] = failure_by_task[record["task_id"]]
    context = build_runtime_context()
    examples = build_conversation_examples(
        records, context=context, tokenizer=tokenizer
    )
    resolved_required_tasks = _required_tasks(examples, required_task_ids)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "records": len(records),
        "strict_success_records": sum(
            bool(record.get("strict_success")) for record in records
        ),
        "available_examples": len(examples),
        "available_assistant_tokens": sum(
            int(example["assistant_token_count"]) for example in examples
        ),
        "available_total_context_tokens": sum(
            int(example["total_token_count"]) for example in examples
        ),
        "available_failure_buckets": dict(
            sorted(
                (
                    bucket,
                    sum(example["failure_bucket"] == bucket for example in examples),
                )
                for bucket in FAILURE_WEIGHTS
            )
        ),
        "target_assistant_tokens": target_tokens,
        "max_examples_per_task": MAX_EXAMPLES_PER_TASK,
        "required_task_count": len(resolved_required_tasks),
        "context_sha256": context["sha256"],
        "arms": {},
    }
    coverage_fingerprints = coverage_example_fingerprints(
        examples,
        required_task_ids=resolved_required_tasks,
        seed=seed,
    )
    uniform_selected = select_budget(
        examples,
        strategy="uniform",
        target_tokens=target_tokens,
        seed=seed,
        required_task_ids=resolved_required_tasks,
    )
    uniform_fingerprints = {
        _example_fingerprint(example) for example in uniform_selected
    }
    uniform_second_pass = uniform_fingerprints - coverage_fingerprints
    failure_aware_selected = select_budget(
        examples,
        strategy="failure_aware",
        target_tokens=target_tokens,
        seed=seed,
        required_task_ids=resolved_required_tasks,
        excluded_second_pass_fingerprints=uniform_second_pass,
    )
    selected_by_arm = {
        "uniform": uniform_selected,
        "failure_aware": failure_aware_selected,
    }
    for strategy, selected in selected_by_arm.items():
        destination = output_dir / f"sft_{strategy}.jsonl"
        write_examples(selected, destination)
        parquet_destination = output_dir / f"sft_{strategy}.parquet"
        write_parquet(selected, parquet_destination)
        summary["arms"][strategy] = {
            **_selection_summary(selected, required_task_ids),
            "failure_buckets": dict(
                sorted(
                    Counter(
                        example["failure_bucket"] for example in selected
                    ).items()
                )
            ),
            "path": str(destination),
            "parquet_path": str(parquet_destination),
        }

    uniform_fingerprints = {
        _example_fingerprint(example) for example in uniform_selected
    }
    failure_aware_fingerprints = {
        _example_fingerprint(example) for example in failure_aware_selected
    }
    shared_fingerprints = uniform_fingerprints & failure_aware_fingerprints
    shared_tokens = sum(
        int(example["assistant_token_count"])
        for example in uniform_selected
        if _example_fingerprint(example) in shared_fingerprints
    )
    uniform_tokens = sum(
        int(example["assistant_token_count"]) for example in uniform_selected
    )
    failure_aware_tokens = sum(
        int(example["assistant_token_count"])
        for example in failure_aware_selected
    )
    summary["cross_arm"] = {
        "shared_trajectories": len(shared_fingerprints),
        "uniform_only_trajectories": len(
            uniform_fingerprints - failure_aware_fingerprints
        ),
        "failure_aware_only_trajectories": len(
            failure_aware_fingerprints - uniform_fingerprints
        ),
        "shared_assistant_tokens": shared_tokens,
        "shared_token_share_uniform": (
            shared_tokens / uniform_tokens if uniform_tokens else 0.0
        ),
        "shared_token_share_failure_aware": (
            shared_tokens / failure_aware_tokens if failure_aware_tokens else 0.0
        ),
        "uniform_second_pass_excluded_from_failure_aware": len(
            uniform_second_pass
        ),
    }

    summary_path = output_dir / "dataset_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
