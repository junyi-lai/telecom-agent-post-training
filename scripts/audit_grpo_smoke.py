from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-log", type=Path, required=True)
    parser.add_argument("--deepseek-log", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-episodes", type=int, default=64)
    parser.add_argument("--minimum-groups", type=int, default=8)
    parser.add_argument("--minimum-group-size", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=32768)
    # A valid smoke contains at least 64 trajectories, and each trajectory may
    # span multiple user-simulator turns. Allow one retry event per trajectory
    # while still rejecting unbounded retry storms.
    parser.add_argument("--max-retries", type=int, default=64)
    parser.add_argument("--max-rate-limit-events", type=int, default=32)
    parser.add_argument("--max-timeout-events", type=int, default=0)
    args = parser.parse_args()

    episodes = _load_jsonl(args.context_log)
    if len(episodes) < args.minimum_episodes:
        raise ValueError(
            f"Expected at least {args.minimum_episodes} agent episodes, "
            f"found {len(episodes)}"
        )

    groups: dict[str, list[float]] = defaultdict(list)
    max_prompt_tokens = 0
    for episode in episodes:
        reward = float(episode["reward_score"])
        if not math.isfinite(reward):
            raise ValueError(f"Non-finite reward for task {episode.get('task_id')}")
        lengths = [int(value) for value in episode.get("prompt_token_lengths", [])]
        if not lengths:
            raise ValueError(f"No token lengths for task {episode.get('task_id')}")
        if any(current < previous for previous, current in zip(lengths, lengths[1:])):
            raise ValueError(
                f"Context length decreased for task {episode.get('task_id')}"
            )
        if max(lengths) > args.max_prompt_tokens:
            raise ValueError(
                f"Prompt exceeded {args.max_prompt_tokens} tokens for "
                f"task {episode.get('task_id')}: {max(lengths)}"
            )
        max_prompt_tokens = max(max_prompt_tokens, max(lengths))
        groups[str(episode["task_id"])].append(reward)

    group_ranges = {
        task_id: max(rewards) - min(rewards)
        for task_id, rewards in groups.items()
    }
    complete_groups = {
        task_id: rewards
        for task_id, rewards in groups.items()
        if len(rewards) >= args.minimum_group_size
    }
    if len(complete_groups) < args.minimum_groups:
        raise ValueError(
            f"Expected at least {args.minimum_groups} complete task groups, "
            f"found {len(complete_groups)}"
        )
    variable_groups = {
        task_id: group_ranges[task_id]
        for task_id in complete_groups
        if group_ranges[task_id] > 0
    }
    if not variable_groups:
        raise ValueError("No reward variation was observed within any task group")
    if not any(int(episode.get("tool_calls", 0)) > 0 for episode in episodes):
        raise ValueError("No tool call was observed in the smoke trajectories")
    valid_tool_call_episodes = sum(
        int(episode.get("tool_calls", 0)) > 0
        and int(episode.get("invalid_tool_calls", 0)) == 0
        for episode in episodes
    )
    if valid_tool_call_episodes == 0:
        raise ValueError("No episode completed a tool call without an invalid call")

    events = _load_jsonl(args.deepseek_log)
    event_counts = Counter(str(event["event"]) for event in events)
    rate_limit_events = sum(event.get("status_code") == 429 for event in events)
    timeout_events = sum(bool(event.get("timeout")) for event in events)
    retry_events = event_counts["retry"]
    failure_events = event_counts["failure"]
    if rate_limit_events > args.max_rate_limit_events:
        raise ValueError(f"DeepSeek 429 events: {rate_limit_events}")
    if timeout_events > args.max_timeout_events:
        raise ValueError(f"DeepSeek timeout events: {timeout_events}")
    if retry_events > args.max_retries:
        raise ValueError(f"DeepSeek retry events: {retry_events}")
    if failure_events:
        raise ValueError(f"DeepSeek final failure events: {failure_events}")
    train_log = args.train_log.read_text(encoding="utf-8", errors="replace")
    nan_markers = re.findall(r"(?i)\b(?:nan|inf(?:inity)?)\b", train_log)
    if nan_markers:
        raise ValueError(f"Found non-finite training log markers: {nan_markers[:5]}")

    result = {
        "agent_episodes": len(episodes),
        "task_groups": len(complete_groups),
        "partial_or_evicted_task_groups": len(groups) - len(complete_groups),
        "task_group_sizes": dict(
            sorted(
                Counter(
                    len(values) for values in complete_groups.values()
                ).items()
            )
        ),
        "variable_reward_groups": len(variable_groups),
        "max_reward_range": max(group_ranges.values()),
        "tool_call_episodes": sum(
            int(episode.get("tool_calls", 0)) > 0 for episode in episodes
        ),
        "valid_tool_call_episodes": valid_tool_call_episodes,
        "max_prompt_tokens": max_prompt_tokens,
        "nonfinite_rewards": 0,
        "nonfinite_training_log_markers": 0,
        "deepseek_event_counts": dict(event_counts),
        "deepseek_429_events": rate_limit_events,
        "deepseek_timeout_events": timeout_events,
        "deepseek_retry_events": retry_events,
        "deepseek_failure_events": failure_events,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
