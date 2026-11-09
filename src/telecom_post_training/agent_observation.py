from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tau2.data_model.message import AssistantMessage, MultiToolMessage
from tau2.utils.llm_utils import to_litellm_messages


class ObservationHistoryDiverged(ValueError):
    """Raised when a tau2 observation is not an extension of its prior history."""


_AUDIT_LOCK = threading.Lock()
_SEMANTIC_MESSAGE_FIELDS = {
    "id",
    "role",
    "content",
    "tool_calls",
    "tool_call_id",
    "requestor",
    "error",
    "name",
}


def message_items(message: Any) -> list[Any]:
    if isinstance(message, MultiToolMessage):
        return list(message.tool_messages)
    return [message]


def _flatten_observation(observation: list[Any] | None) -> list[Any]:
    messages = []
    for message in observation or []:
        messages.extend(message_items(message))
    return messages


def _message_key(message: Any) -> tuple[str, str]:
    if hasattr(message, "model_dump"):
        payload = message.model_dump(
            mode="json",
            include=_SEMANTIC_MESSAGE_FIELDS,
            exclude_none=True,
        )
    else:
        payload = {
            key: value
            for key, value in message.items()
            if key in _SEMANTIC_MESSAGE_FIELDS and value is not None
        }
    return (
        type(message).__name__,
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def new_context_messages(
    previous_observation: list[Any] | None,
    current_observation: list[Any] | None,
) -> list[dict]:
    """Return only the history suffix added since the previous observation."""
    previous = _flatten_observation(previous_observation)
    current = _flatten_observation(current_observation)
    previous_keys = [_message_key(message) for message in previous]
    current_keys = [_message_key(message) for message in current]

    common_prefix = 0
    while (
        common_prefix < len(previous_keys)
        and common_prefix < len(current_keys)
        and previous_keys[common_prefix] == current_keys[common_prefix]
    ):
        common_prefix += 1
    if common_prefix != len(previous_keys):
        # tau2 may cap the observation to a rolling window. In that case the
        # current history starts with a suffix of the previous history.
        max_overlap = min(len(previous_keys), len(current_keys))
        overlap = 0
        for candidate in range(max_overlap, 0, -1):
            if previous_keys[-candidate:] == current_keys[:candidate]:
                overlap = candidate
                break
        if overlap == 0:
            raise ObservationHistoryDiverged(
                "tau2 observation history is neither an extension nor a rolling "
                "window of the prior observation"
            )
        common_prefix = overlap

    context = []
    for message in current[common_prefix:]:
        # Generated assistant turns are appended directly from the policy output.
        if isinstance(message, AssistantMessage):
            continue
        converted = to_litellm_messages([message])
        if not converted:
            raise ValueError(
                f"Unsupported observation message type: {type(message).__name__}"
            )
        context.extend(converted)
    return context


def append_context_audit(record: dict[str, Any]) -> None:
    path_value = os.getenv("TAU2_CONTEXT_AUDIT_PATH")
    if not path_value:
        return
    path = Path(path_value).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        **record,
    }
    with _AUDIT_LOCK, path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
