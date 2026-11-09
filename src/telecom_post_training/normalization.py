from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def iter_simulations(path: Path) -> list[dict]:
    """Load either a monolithic results.json or a directory result bundle."""
    results_path = path
    simulations_dir: Path | None = None
    if path.is_dir():
        results_path = path / "results.json"
        simulations_dir = path / "simulations"

    if not results_path.exists():
        raise FileNotFoundError(f"Missing tau2 result file: {results_path}")
    data = _read_json(results_path)
    if not isinstance(data, dict):
        raise TypeError(f"Unexpected results root in {results_path}")

    if simulations_dir and simulations_dir.is_dir():
        return [
            _read_json(simulation_path)
            for simulation_path in sorted(simulations_dir.glob("*.json"))
        ]
    simulations = data.get("simulations", [])
    if not isinstance(simulations, list):
        raise TypeError("Expected simulations to be a list")
    return simulations


def _normalize_message(message: dict) -> dict:
    role = message.get("role")
    normalized = {
        "role": role,
        "content": message.get("content"),
    }
    if message.get("tool_calls"):
        normalized["tool_calls"] = message["tool_calls"]
    raw_data = message.get("raw_data") or {}
    if raw_data.get("finish_reason") == "length":
        normalized["output_truncated"] = True
    if role == "tool":
        normalized["tool_call_id"] = message.get("id")
        normalized["requestor"] = message.get("requestor")
        normalized["error"] = bool(message.get("error", False))
        normalized["execution_status"] = message.get("execution_status", "executed")
    return normalized


def _tool_calls(messages: list[dict]) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            name = call.get("name", "")
            arguments = json.dumps(
                call.get("arguments") or {},
                sort_keys=True,
                separators=(",", ":"),
            )
            calls.append((name, arguments))
    return calls


def _max_consecutive_repeat(calls: list[tuple[str, str]]) -> int:
    longest = 0
    current = 0
    previous: tuple[str, str] | None = None
    for call in calls:
        if call == previous:
            current += 1
        else:
            current = 1
            previous = call
        longest = max(longest, current)
    return longest


def classify_failure(
    simulation: dict,
    normalized_messages: list[dict],
) -> list[str]:
    reward_info = simulation.get("reward_info") or {}
    reward = reward_info.get("reward")
    strict_success = isinstance(reward, (int, float)) and float(reward) >= 1.0 - 1e-9
    if strict_success:
        return ["success"]

    tags: set[str] = set()
    termination = simulation.get("termination_reason")
    if termination in {
        "infrastructure_error",
        "timeout",
        "unexpected_error",
        "context_window_exceeded",
    }:
        tags.add("infra_error")
    if termination == "max_steps":
        tags.add("max_steps")
    if termination == "agent_error":
        tags.add("agent_error")
    if termination == "agent_stop":
        tags.add("premature_exit")

    if any(
        message.get("role") == "tool" and message.get("error")
        for message in normalized_messages
    ):
        tags.add("invalid_tool_call")
    if any(
        message.get("role") == "assistant" and message.get("output_truncated")
        for message in normalized_messages
    ):
        tags.add("response_truncated")

    calls = _tool_calls(normalized_messages)
    if _max_consecutive_repeat(calls) >= 2:
        tags.add("repetition")

    action_checks = reward_info.get("action_checks") or []
    if any(not check.get("action_match", False) for check in action_checks):
        tags.add("wrong_action_or_policy")

    env_assertions = reward_info.get("env_assertions") or []
    if any(not check.get("met", False) for check in env_assertions):
        tags.add("incomplete_state")

    db_check = reward_info.get("db_check")
    if db_check and not db_check.get("db_match", False):
        tags.add("wrong_final_state")

    if not tags:
        tags.add("generic_failure")
    return sorted(tags)


def normalize_simulation(
    simulation: dict,
    *,
    split: str,
    model_revision: str,
    user_simulator_model: str,
    source_role: str = "student",
    teacher_model: str | None = None,
) -> dict:
    messages = simulation.get("messages") or []
    normalized_messages = [_normalize_message(message) for message in messages]
    reward_info = simulation.get("reward_info") or {}
    strict_success = float(reward_info.get("reward") or 0.0) >= 1.0 - 1e-9
    calls = _tool_calls(normalized_messages)

    return {
        "schema_version": 1,
        "simulation_id": simulation.get("id"),
        "task_id": simulation.get("task_id"),
        "split": split,
        "trial": simulation.get("trial"),
        "seed": simulation.get("seed"),
        "termination_reason": simulation.get("termination_reason"),
        "duration_seconds": simulation.get("duration"),
        "agent_cost": simulation.get("agent_cost"),
        "user_cost": simulation.get("user_cost"),
        "strict_success": strict_success,
        "reward_info": reward_info or None,
        "failure_tags": classify_failure(simulation, normalized_messages),
        "truncated_outputs": sum(
            bool(message.get("output_truncated")) for message in normalized_messages
        ),
        "turn_count": sum(
            1 for message in normalized_messages if message.get("role") == "assistant"
        ),
        "tool_call_count": len(calls),
        "max_consecutive_repeat": _max_consecutive_repeat(calls),
        "messages": normalized_messages,
        "provenance": {
            "source_role": source_role,
            "student_model": (simulation.get("info") or {}).get("agent_info"),
            "user_simulator_model": user_simulator_model,
            "student_model_revision": (
                model_revision if source_role == "student" else None
            ),
            "offline_teacher_model": (
                teacher_model if source_role == "offline_teacher" else None
            ),
        },
    }


def write_jsonl(records: list[dict], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
