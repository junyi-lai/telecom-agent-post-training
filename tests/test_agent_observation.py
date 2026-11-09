from copy import deepcopy

import pytest
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)

from telecom_post_training.agent_observation import (
    ObservationHistoryDiverged,
    new_context_messages,
)


def test_deepcopy_replay_adds_each_history_message_once() -> None:
    first_user = UserMessage.text("hello")
    assistant = AssistantMessage.text(
        "checking",
        tool_calls=[
            ToolCall(
                id="agent-call",
                name="check_network_status",
                arguments={},
                requestor="assistant",
            )
        ],
    )
    agent_tool_result = ToolMessage(
        id="agent-call",
        role="tool",
        content="network is offline",
        requestor="assistant",
    )
    second_user = UserMessage.text("please try another fix")
    user_tool_result = MultiToolMessage(
        role="tool",
        tool_messages=[
            ToolMessage(
                id="user-call",
                role="tool",
                content="customer confirmed Wi-Fi is off",
                requestor="user",
            )
        ],
    )
    history = [
        first_user,
        assistant,
        agent_tool_result,
        second_user,
        user_tool_result,
    ]
    observations = [
        deepcopy(history[: length]) for length in (1, 4, 5)
    ]
    assert id(first_user) != id(observations[-1][0])

    context = []
    previous = []
    for observation in observations:
        context.extend(new_context_messages(previous, observation))
        previous = observation

    assert [message["content"] for message in context].count("hello") == 1
    assert (
        [message["content"] for message in context].count("network is offline")
        == 1
    )
    assert (
        [message["content"] for message in context].count(
            "customer confirmed Wi-Fi is off"
        )
        == 1
    )
    assert "checking" not in [message["content"] for message in context]
    assert [message["role"] for message in context] == [
        "user",
        "tool",
        "user",
        "tool",
    ]


def test_observation_must_extend_previous_full_history() -> None:
    previous = [UserMessage.text("first")]
    current = [UserMessage.text("replacement")]
    with pytest.raises(ObservationHistoryDiverged):
        new_context_messages(previous, current)


def test_volatile_message_metadata_does_not_break_common_prefix() -> None:
    previous = [
        UserMessage(
            role="user",
            content="hello",
            timestamp="2026-01-01T00:00:00Z",
            cost=0.0,
        )
    ]
    current = [
        UserMessage(
            role="user",
            content="hello",
            timestamp="2026-01-01T00:00:01Z",
            cost=0.5,
        )
    ]
    assert new_context_messages(previous, current) == []


def test_rolling_observation_window_appends_only_new_suffix() -> None:
    first = UserMessage.text("first")
    second = UserMessage.text("second")
    third = UserMessage.text("third")
    fourth = UserMessage.text("fourth")

    assert new_context_messages([first, second, third], [second, third, fourth]) == [
        {"role": "user", "content": "fourth"}
    ]
