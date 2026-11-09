import asyncio
import threading

from telecom_post_training import rollout_runtime
from telecom_post_training.agent_environment import TelecomAgentGymEnv
from telecom_post_training.rollout_runtime import ProcessRolloutLimiter


class _Participant:
    def __init__(self) -> None:
        self.stop_calls = 0

    def stop(self) -> None:
        self.stop_calls += 1


def test_environment_close_stops_participants_and_thread() -> None:
    env = TelecomAgentGymEnv(domain="telecom", task_id="unused")
    agent = _Participant()
    user = _Participant()
    started = threading.Event()

    def wait_for_close() -> None:
        started.set()
        env._simulation_done.wait()

    thread = threading.Thread(target=wait_for_close)
    env._agent = agent
    env._user = user
    env._orchestrator_thread = thread
    thread.start()
    started.wait(timeout=1)

    env.close()

    assert agent.stop_calls == 1
    assert user.stop_calls == 1
    assert not thread.is_alive()
    assert env._simulation_done.is_set()


def test_rollout_semaphore_limits_each_worker_process(monkeypatch) -> None:
    monkeypatch.setenv("TAU2_MAX_ACTIVE_ROLLOUTS_PER_WORKER", "2")

    async def scenario() -> int:
        semaphore = ProcessRolloutLimiter.semaphore(2)
        active = 0
        maximum = 0

        async def run_one() -> None:
            nonlocal active, maximum
            async with semaphore:
                active += 1
                maximum = max(maximum, active)
                await asyncio.sleep(0.01)
                active -= 1

        await asyncio.gather(*(run_one() for _ in range(8)))
        return maximum

    assert asyncio.run(scenario()) == 2


def test_rollout_memory_release_runs_gc_and_malloc_trim(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        rollout_runtime.gc,
        "collect",
        lambda: calls.append("gc"),
    )
    monkeypatch.setattr(
        rollout_runtime,
        "_MALLOC_TRIM",
        lambda value: calls.append(f"trim:{value}") or 1,
    )

    rollout_runtime.release_process_memory()

    assert calls == ["gc", "trim:0"]
