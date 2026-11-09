from telecom_post_training.rollout_runtime import round_robin_slice


def test_agent_loop_workers_rotate_for_single_prompt_refills() -> None:
    workers = list(range(4))
    cursor = 0

    selected, cursor = round_robin_slice(workers, 1, cursor)
    assert selected == [0]
    selected, cursor = round_robin_slice(workers, 1, cursor)
    assert selected == [1]
    selected, cursor = round_robin_slice(workers, 2, cursor)
    assert selected == [2, 3]
    selected, cursor = round_robin_slice(workers, 3, cursor)
    assert selected == [0, 1, 2]
    selected, cursor = round_robin_slice(workers, 0, cursor)
    assert selected == []
