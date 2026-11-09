from telecom_post_training.agent_environment import TelecomAgentGymEnv


def test_training_environment_uses_complete_telecom_task_catalog() -> None:
    env = TelecomAgentGymEnv(domain="telecom", task_id="unused")
    config = env._run_config(seed=20261103)

    assert config.task_set_name == "telecom"
    assert config.task_split_name == "full"
    assert config.seed == 20261103
