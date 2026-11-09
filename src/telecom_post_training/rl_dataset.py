from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from telecom_post_training.manifest import parse_task_id

FAILURE_PRIORITY = {
    "wrong_action_or_policy": 4.0,
    "invalid_tool_call": 3.0,
    "incomplete_state": 3.0,
    "repetition": 2.0,
    "premature_exit": 2.0,
    "response_truncated": 1.5,
}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def task_failure_stats(records: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        task_id = record.get("task_id")
        if task_id:
            grouped[task_id].append(record)

    result = {}
    for task_id, task_records in grouped.items():
        tag_counts: Counter[str] = Counter()
        for record in task_records:
            tag_counts.update(record.get("failure_tags") or [])
        failures = sum(not record.get("strict_success") for record in task_records)
        result[task_id] = {
            "count": len(task_records),
            "failure_rate": failures / len(task_records),
            "failure_tags": dict(tag_counts),
        }
    return result


def failure_aware_weight(task_id: str, stats: dict | None) -> float:
    identity = parse_task_id(task_id)
    weight = 1.0 + 0.15 * (len(identity.faults) - 1)
    if not stats:
        return weight
    weight += 2.0 * float(stats["failure_rate"])
    for tag, count in stats["failure_tags"].items():
        weight += FAILURE_PRIORITY.get(tag, 0.5) * min(int(count), 4)
    return max(weight, 1.0)


def build_rl_prompts(
    manifest: dict,
    context: dict,
    normalized_records: list[dict],
    *,
    output_dir: Path,
    seed: int,
    task_split: str = "train_pool",
) -> dict:
    stats = task_failure_stats(normalized_records)
    task_ids = list(manifest["splits"][task_split])
    system_message = {"role": "system", "content": context["system_prompt"]}

    def row(task_id: str, sampling_strategy: str, weight: float) -> dict:
        return {
            "prompt": [system_message],
            "tools": context["tools"],
            "agent_name": "tau2_telecom",
            "data_source": "tau2_telecom",
            "extra_info": {
                "task_id": task_id,
                "seed": seed,
                "sampling_strategy": sampling_strategy,
                "sampling_weight": weight,
            },
        }

    uniform_rows = [row(task_id, "uniform", 1.0) for task_id in task_ids]
    weights = [
        failure_aware_weight(task_id, stats.get(task_id)) for task_id in task_ids
    ]
    rng = random.Random(seed)
    coverage_rows = [
        row(
            task_id,
            "failure_aware",
            failure_aware_weight(task_id, stats.get(task_id)),
        )
        for task_id in task_ids
    ]
    weighted_extra_rows = [
        row(
            task_id,
            "failure_aware",
            failure_aware_weight(task_id, stats.get(task_id)),
        )
        for task_id in rng.choices(task_ids, weights=weights, k=len(task_ids))
    ]
    failure_rows = coverage_rows + weighted_extra_rows

    uniform_occurrences = Counter(
        row["extra_info"]["task_id"] for row in uniform_rows
    )
    failure_occurrences = Counter(
        row["extra_info"]["task_id"] for row in failure_rows
    )
    if set(uniform_occurrences) != set(task_ids):
        raise ValueError("Uniform RL prompts do not cover every task")
    if set(failure_occurrences) != set(task_ids):
        raise ValueError("Failure-aware RL prompts do not cover every task")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, rows in (
        ("uniform", uniform_rows),
        ("failure_aware", failure_rows),
    ):
        destination = output_dir / f"rl_{name}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), destination)
        paths[name] = str(destination)

    summary = {
        "task_split": task_split,
        "tasks": len(task_ids),
        "tasks_with_baseline_stats": len(stats),
        "uniform_rows": len(uniform_rows),
        "failure_aware_rows": len(failure_rows),
        "uniform_unique_tasks": len(uniform_occurrences),
        "failure_aware_unique_tasks": len(failure_occurrences),
        "uniform_task_occurrence_distribution": dict(
            sorted(Counter(uniform_occurrences.values()).items())
        ),
        "failure_aware_task_occurrence_distribution": dict(
            sorted(Counter(failure_occurrences.values()).items())
        ),
        "failure_aware_extra_rows": len(weighted_extra_rows),
        "failure_aware_task_occurrences": dict(
            sorted(failure_occurrences.items())
        ),
        "paths": paths,
    }
    (output_dir / "rl_dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
