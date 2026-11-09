from telecom_post_training.normalization import classify_failure, normalize_simulation


def _simulation() -> dict:
    return {
        "id": "sim-1",
        "task_id": "task-1",
        "trial": 0,
        "seed": 42,
        "termination_reason": "agent_stop",
        "duration": 1.5,
        "agent_cost": 0.0,
        "user_cost": 0.0,
        "reward_info": {
            "reward": 0.0,
            "db_check": {"db_match": False, "db_reward": 0.0},
            "env_assertions": [
                {
                    "env_assertion": {},
                    "met": False,
                    "reward": 0.0,
                }
            ],
            "action_checks": [],
        },
        "messages": [
            {"role": "user", "content": "My phone has no service"},
            {
                "role": "assistant",
                "content": None,
                "raw_data": {"finish_reason": "length"},
                "tool_calls": [
                    {
                        "id": "call-1",
                        "name": "check_status_bar",
                        "arguments": {},
                    }
                ],
            },
            {
                "role": "tool",
                "id": "call-1",
                "requestor": "assistant",
                "content": "No Signal",
                "error": False,
                "execution_status": "executed",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-2",
                        "name": "check_status_bar",
                        "arguments": {},
                    }
                ],
            },
        ],
    }


def test_failure_classifier_uses_trace_signals() -> None:
    simulation = _simulation()
    messages = [
        {"role": "user", "content": "My phone has no service"},
        {
            "role": "assistant",
            "tool_calls": [{"name": "check_status_bar", "arguments": {}}],
        },
        {
            "role": "tool",
            "error": False,
            "execution_status": "executed",
        },
        {
            "role": "assistant",
            "tool_calls": [{"name": "check_status_bar", "arguments": {}}],
        },
    ]
    assert "repetition" in classify_failure(simulation, messages)


def test_normalization_preserves_tool_metadata() -> None:
    record = normalize_simulation(
        _simulation(),
        split="train",
        model_revision="model-sha",
        user_simulator_model="deepseek-test",
    )
    assert record["strict_success"] is False
    assert record["turn_count"] == 2
    assert record["tool_call_count"] == 2
    assert record["truncated_outputs"] == 1
    assert "premature_exit" in record["failure_tags"]
    assert "response_truncated" in record["failure_tags"]
    assert record["messages"][2]["tool_call_id"] == "call-1"


def test_teacher_provenance_does_not_mislabel_student_revision() -> None:
    record = normalize_simulation(
        _simulation(),
        split="teacher",
        model_revision="student-sha",
        user_simulator_model="deepseek-flash",
        source_role="offline_teacher",
        teacher_model="deepseek-flash",
    )
    provenance = record["provenance"]
    assert provenance["source_role"] == "offline_teacher"
    assert provenance["student_model_revision"] is None
    assert provenance["offline_teacher_model"] == "deepseek-flash"
