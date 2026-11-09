import pytest

from telecom_post_training.agent_reward import compute_agent_reward


def test_success_reward_is_positive_and_clipped() -> None:
    result = compute_agent_reward(
        reward_info={
            "reward": 1.0,
            "action_checks": [{"action_match": True}],
            "env_assertions": [{"met": True}],
        },
        termination_reason="user_stop",
        invalid_tool_calls=0,
        repeated_actions=0,
        strict_success=True,
    )
    assert result.total == 1.1
    assert result.policy_violation is False


def test_failure_penalties_are_deterministic() -> None:
    result = compute_agent_reward(
        reward_info={
            "reward": 0.0,
            "action_checks": [{"action_match": False}],
            "env_assertions": [{"met": True}, {"met": False}],
        },
        termination_reason="agent_stop",
        invalid_tool_calls=2,
        repeated_actions=3,
        strict_success=False,
    )
    assert result.policy_violation is True
    assert result.premature_exit is True
    assert result.progress == 0.5
    assert result.total == pytest.approx(-0.47)
