from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _records_by_task(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for index, record in enumerate(records):
        task_id = record.get("task_id") or record.get("simulation_id") or str(index)
        grouped.setdefault(str(task_id), []).append(record)
    return grouped


def _task_success_rates(records: list[dict]) -> dict[str, float]:
    return {
        task_id: sum(bool(record.get("strict_success")) for record in task_records)
        / len(task_records)
        for task_id, task_records in _records_by_task(records).items()
    }


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * probability)
    return ordered[index]


def task_cluster_summary(
    records: list[dict],
    *,
    n_bootstrap: int = 2_000,
    seed: int = 20261103,
) -> dict:
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    grouped = _records_by_task(records)
    task_rates = _task_success_rates(records)
    rates = list(task_rates.values())
    rng = random.Random(seed)
    bootstrap_means = []
    if rates:
        for _ in range(n_bootstrap):
            sample = [rates[rng.randrange(len(rates))] for _ in rates]
            bootstrap_means.append(sum(sample) / len(sample))
    else:
        bootstrap_means.append(0.0)
    episodes_per_task = [len(values) for values in grouped.values()]
    return {
        "task_clusters": len(task_rates),
        "task_success_rate_mean": sum(rates) / len(rates) if rates else 0.0,
        "task_success_rate_bootstrap_95": [
            _percentile(bootstrap_means, 0.025),
            _percentile(bootstrap_means, 0.975),
        ],
        "task_outcomes": {
            "all_fail": sum(rate == 0.0 for rate in rates),
            "partial": sum(0.0 < rate < 1.0 for rate in rates),
            "all_success": sum(rate == 1.0 for rate in rates),
        },
        "episodes_per_task_min": min(episodes_per_task) if episodes_per_task else 0,
        "episodes_per_task_max": max(episodes_per_task) if episodes_per_task else 0,
        "bootstrap_samples": n_bootstrap,
        "bootstrap_seed": seed,
    }


def paired_task_bootstrap(
    left_records: list[dict],
    right_records: list[dict],
    *,
    n_bootstrap: int = 10_000,
    seed: int = 20261103,
) -> dict:
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    left_rates = _task_success_rates(left_records)
    right_rates = _task_success_rates(right_records)
    left_tasks = set(left_rates)
    right_tasks = set(right_rates)
    if left_tasks != right_tasks:
        missing_left = sorted(right_tasks - left_tasks)
        missing_right = sorted(left_tasks - right_tasks)
        raise ValueError(
            "Paired comparison requires identical task clusters; "
            f"missing_left={missing_left}, missing_right={missing_right}"
        )

    task_ids = sorted(left_tasks)
    differences = [left_rates[task_id] - right_rates[task_id] for task_id in task_ids]
    observed_difference = (
        sum(differences) / len(differences) if differences else 0.0
    )
    rng = random.Random(seed)
    bootstrap_differences = []
    if differences:
        for _ in range(n_bootstrap):
            sampled = [
                differences[rng.randrange(len(differences))] for _ in differences
            ]
            bootstrap_differences.append(sum(sampled) / len(sampled))
    else:
        bootstrap_differences.append(0.0)

    low = _percentile(bootstrap_differences, 0.025)
    high = _percentile(bootstrap_differences, 0.975)
    return {
        "task_clusters": len(task_ids),
        "left_episode_success_rate": (
            sum(bool(record.get("strict_success")) for record in left_records)
            / len(left_records)
            if left_records
            else 0.0
        ),
        "right_episode_success_rate": (
            sum(bool(record.get("strict_success")) for record in right_records)
            / len(right_records)
            if right_records
            else 0.0
        ),
        "left_task_success_rate_mean": (
            sum(left_rates.values()) / len(left_rates) if left_rates else 0.0
        ),
        "right_task_success_rate_mean": (
            sum(right_rates.values()) / len(right_rates) if right_rates else 0.0
        ),
        "absolute_difference": observed_difference,
        "absolute_difference_percentage_points": 100.0 * observed_difference,
        "paired_bootstrap_95": [low, high],
        "paired_bootstrap_95_percentage_points": [100.0 * low, 100.0 * high],
        "probability_left_better": (
            sum(value > 0.0 for value in bootstrap_differences)
            / n_bootstrap
            if bootstrap_differences
            else 0.0
        ),
        "bootstrap_samples": n_bootstrap,
        "bootstrap_seed": seed,
        "task_level_differences": {
            task_id: left_rates[task_id] - right_rates[task_id]
            for task_id in task_ids
        },
    }


def summarize(records: list[dict]) -> dict:
    total = len(records)
    successes = sum(bool(record.get("strict_success")) for record in records)
    low, high = wilson_interval(successes, total)
    terminations = Counter(str(record.get("termination_reason")) for record in records)
    user_api_cost = sum(record.get("user_cost") or 0.0 for record in records)
    agent_api_cost = sum(record.get("agent_cost") or 0.0 for record in records)
    failures = Counter(
        tag
        for record in records
        if not record.get("strict_success")
        for tag in record.get("failure_tags") or []
    )
    task_summary = task_cluster_summary(records)
    return {
        "episodes": total,
        "successes": successes,
        "success_rate": successes / total if total else 0.0,
        "success_rate_wilson_95": [low, high],
        "terminations": dict(terminations),
        "failures": dict(failures),
        "average_turns": (
            sum(record.get("turn_count") or 0 for record in records) / total
            if total
            else 0.0
        ),
        "average_tool_calls": (
            sum(record.get("tool_call_count") or 0 for record in records) / total
            if total
            else 0.0
        ),
        "average_duration_seconds": (
            sum(record.get("duration_seconds") or 0.0 for record in records) / total
            if total
            else 0.0
        ),
        "user_api_cost": user_api_cost,
        "agent_api_cost": agent_api_cost,
        "total_api_cost": user_api_cost + agent_api_cost,
        **task_summary,
    }


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def render_markdown(summary: dict) -> str:
    low, high = summary["success_rate_wilson_95"]
    task_low, task_high = summary["task_success_rate_bootstrap_95"]
    task_outcomes = summary["task_outcomes"]
    failure_rows = "\n".join(
        f"| `{tag}` | {count} |" for tag, count in sorted(summary["failures"].items())
    )
    return f"""# 运行摘要

| 指标 | 数值 |
| --- | ---: |
| 回合数 | {summary["episodes"]} |
| 严格成功数 | {summary["successes"]} |
| 成功率 | {summary["success_rate"]:.4f} |
| 95% Wilson 区间 | [{low:.4f}, {high:.4f}] |
| 平均助手轮数 | {summary["average_turns"]:.2f} |
| 平均工具调用次数 | {summary["average_tool_calls"]:.2f} |
| 平均耗时 | {summary["average_duration_seconds"]:.2f} 秒 |
| 用户模拟器成本 | ${summary["user_api_cost"]:.6f} |
| Agent API 成本 | ${summary["agent_api_cost"]:.6f} |
| 总 API 成本 | ${summary["total_api_cost"]:.6f} |
| 任务簇数 | {summary["task_clusters"]} |
| 任务等权成功率 | {summary["task_success_rate_mean"]:.4f} |
| 任务级 bootstrap 95% 区间 | [{task_low:.4f}, {task_high:.4f}] |
| 全失败 / 部分成功 / 全成功任务数 | {task_outcomes["all_fail"]} / {task_outcomes["partial"]} / {task_outcomes["all_success"]} |
| 每任务回合数范围 | {summary["episodes_per_task_min"]}-{summary["episodes_per_task_max"]} |

## 失败标签

| 标签 | 数量 |
| --- | ---: |
{failure_rows}
"""


def render_paired_comparison(result: dict) -> str:
    low, high = result["paired_bootstrap_95_percentage_points"]
    return f"""# 任务簇配对比较

| 指标 | 数值 |
| --- | ---: |
| 任务簇数 | {result["task_clusters"]} |
| 左侧回合成功率 | {result["left_episode_success_rate"]:.4f} |
| 右侧回合成功率 | {result["right_episode_success_rate"]:.4f} |
| 左侧任务等权成功率 | {result["left_task_success_rate_mean"]:.4f} |
| 右侧任务等权成功率 | {result["right_task_success_rate_mean"]:.4f} |
| 配对绝对差 | {result["absolute_difference_percentage_points"]:+.2f} 个百分点 |
| 配对 bootstrap 95% 区间 | [{low:+.2f}, {high:+.2f}] 个百分点 |
| 左侧更优的 bootstrap 概率 | {result["probability_left_better"]:.4f} |

## 统计协议

每个任务 ID 是一个独立的重采样簇。先对每个任务跨固定种子的成功率求均值，
再对 40 个任务 ID 做有放回配对 bootstrap，并将同一组抽样索引同时应用于两个
实验组。单条轨迹不视为独立样本。

配置：{result["bootstrap_samples"]} 次 bootstrap 采样，随机种子
`{result["bootstrap_seed"]}`。
"""
