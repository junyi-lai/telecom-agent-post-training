from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardBreakdown:
    task_reward: float
    progress: float
    policy_violation: bool
    invalid_tool_calls: int
    repeated_actions: int
    premature_exit: bool
    total: float


def compute_agent_reward(
    *,
    reward_info: dict,
    termination_reason: str | None,
    invalid_tool_calls: int,
    repeated_actions: int,
    strict_success: bool,
) -> RewardBreakdown:
    action_checks = reward_info.get("action_checks") or []
    env_assertions = reward_info.get("env_assertions") or []
    policy_violation = any(
        not check.get("action_match", False) for check in action_checks
    )
    if env_assertions:
        progress = sum(bool(check.get("met", False)) for check in env_assertions) / len(
            env_assertions
        )
    else:
        progress = 1.0 if strict_success else 0.0
    premature_exit = termination_reason == "agent_stop" and not strict_success

    total = (
        (1.0 if strict_success else 0.0)
        + 0.10 * progress
        - (0.25 if policy_violation else 0.0)
        - 0.02 * min(max(invalid_tool_calls, 0), 5)
        - 0.01 * min(max(repeated_actions, 0), 10)
        - (0.20 if premature_exit else 0.0)
    )
    total = max(-0.50, min(total, 1.10))
    return RewardBreakdown(
        task_reward=float(reward_info.get("reward") or 0.0),
        progress=progress,
        policy_violation=policy_violation,
        invalid_tool_calls=invalid_tool_calls,
        repeated_actions=repeated_actions,
        premature_exit=premature_exit,
        total=total,
    )
