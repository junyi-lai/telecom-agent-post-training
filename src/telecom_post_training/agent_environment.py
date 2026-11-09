from __future__ import annotations

import threading
from typing import Optional

from tau2.data_model.simulation import TextRunConfig
from tau2.gym.gym_agent import GYM_AGENT, GYM_SOLO_AGENT, AgentGymEnv


class TelecomAgentGymEnv(AgentGymEnv):
    """AgentGymEnv variant that exposes the complete telecom task catalog."""

    def _run_config(self, seed: Optional[int] = None) -> TextRunConfig:
        return TextRunConfig(
            domain=self.domain,
            task_set_name=self.domain,
            task_split_name="full",
            language=self.language,
            agent=GYM_SOLO_AGENT if self.solo_mode else GYM_AGENT,
            user="dummy_user" if self.solo_mode else "user_simulator",
            llm_user=self.user_llm,
            llm_args_user=self.user_llm_args,
            max_steps=self.max_steps,
            **({} if seed is None else {"seed": seed}),
            **(
                {}
                if self.communicate_scoring is None
                else {"communicate_scoring": self.communicate_scoring}
            ),
        )

    def close(self) -> None:
        """Release a timed-out rollout instead of leaving its threads alive."""
        self._simulation_done.set()
        for participant_name in ("_agent", "_user"):
            participant = getattr(self, participant_name, None)
            stop = getattr(participant, "stop", None)
            if stop is None:
                continue
            try:
                stop()
            except Exception:
                pass

        thread = getattr(self, "_orchestrator_thread", None)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        super().close()
