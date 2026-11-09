"""veRL AgentLoop for the pinned tau2 telecom environment.

This module requires the training environment with veRL installed.
Keeping the integration in the main package lets the registry config
load it by fully qualified class name.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any
from uuid import uuid4

from tau2.data_model.message import ToolMessage
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

from telecom_post_training.agent_environment import TelecomAgentGymEnv
from telecom_post_training.agent_observation import (
    append_context_audit,
    message_items,
    new_context_messages,
)
from telecom_post_training.agent_reward import compute_agent_reward
from telecom_post_training.deepseek_client import configure_deepseek_client
from telecom_post_training.rollout_runtime import (
    ProcessRolloutLimiter,
    release_process_memory,
)


def _assistant_message(content: str | None, tool_calls: list[Any]) -> dict:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content or "",
    }
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": call.tool_call_id or f"call-{index}",
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.arguments,
                },
            }
            for index, call in enumerate(tool_calls)
        ]
    return message


def _action_payload(content: str | None, tool_calls: list[Any]) -> tuple[str, bool]:
    if not tool_calls:
        return content or "", False
    call = tool_calls[0]
    try:
        arguments = json.loads(call.arguments)
    except (json.JSONDecodeError, TypeError):
        return "", True
    return json.dumps(
        {"name": call.name, "arguments": arguments},
        ensure_ascii=False,
        separators=(",", ":"),
    ), len(tool_calls) > 1


def _maximum_consecutive_repeat(tool_calls: list[tuple[str, str]]) -> int:
    longest = 0
    current = 0
    previous: tuple[str, str] | None = None
    for call in tool_calls:
        if call == previous:
            current += 1
        else:
            previous = call
            current = 1
        longest = max(longest, current)
    return longest


@register("tau2_telecom")
class Tau2TelecomAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_parser = ToolParser.get_tool_parser("qwen3_coder", self.tokenizer)
        self.deepseek_client = configure_deepseek_client()
        self.response_length = self.rollout_config.response_length
        self.max_turns = int(os.getenv("TAU2_ROLLOUT_MAX_TURNS", "16"))
        self.max_tokens_per_turn = int(os.getenv("TAU2_MAX_TOKENS_PER_TURN", "384"))
        self.env_timeout = float(os.getenv("TAU2_ENV_TIMEOUT", "180"))
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
        self.user_llm = model if model.startswith("deepseek/") else f"deepseek/{model}"
        self.user_llm_args = {
            "temperature": float(os.getenv("DEEPSEEK_USER_TEMPERATURE", "0.7")),
            "max_tokens": 256,
            "seed": int(os.getenv("DEEPSEEK_USER_SEED", "42")),
            "num_retries": 0,
            "timeout": self.deepseek_client.config.request_timeout_seconds,
            "extra_body": {"thinking": {"type": "disabled"}},
        }

    @classmethod
    def _rollout_semaphore(cls) -> asyncio.Semaphore:
        limit = int(os.getenv("TAU2_MAX_ACTIVE_ROLLOUTS_PER_WORKER", "2"))
        if limit < 1:
            raise ValueError(
                "TAU2_MAX_ACTIVE_ROLLOUTS_PER_WORKER must be positive"
            )
        return ProcessRolloutLimiter.semaphore(limit)

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        semaphore = self._rollout_semaphore()
        async with semaphore:
            try:
                return await self._run_unbounded(sampling_params, **kwargs)
            finally:
                release_process_memory()

    async def _run_unbounded(
        self,
        sampling_params: dict[str, Any],
        **kwargs,
    ) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        extra_info = kwargs.get("extra_info") or {}
        task_id = extra_info.get("task_id") or kwargs.get("task_id")
        if not task_id:
            raise ValueError("Tau2TelecomAgentLoop requires extra_info.task_id")

        env = TelecomAgentGymEnv(
            domain="telecom",
            task_id=task_id,
            max_steps=int(extra_info.get("max_steps", self.max_turns)),
            user_llm=self.user_llm,
            user_llm_args=self.user_llm_args,
            all_messages_as_observation=True,
        )
        seed = int(extra_info.get("seed", 20261103))
        metrics: dict[str, Any] = {}
        message_observation: list[Any] = []
        tool_call_signatures: list[tuple[str, str]] = []
        seen_tool_message_ids: set[str] = set()
        invalid_tool_calls = 0
        final_reward = 0.0
        termination_reason: str | None = None
        reward_info: dict[str, Any] = {}
        response_logprobs: list[float] = []
        client_start = self.deepseek_client.snapshot()

        try:
            _, info = await asyncio.wait_for(
                asyncio.to_thread(env.reset, seed=seed),
                timeout=self.env_timeout,
            )
            raw_observation = getattr(env._agent, "observation", None)
            message_observation = list(raw_observation or [])
            messages.extend(new_context_messages([], raw_observation))
            tools = [tool.openai_schema for tool in info["tools"]]
            prompt_ids = await self.ct_build_initial_tokens(
                messages,
                tools=tools,
            )
            response_mask: list[int] = []
            assistant_turns = 0
            prompt_token_lengths: list[int] = []

            while assistant_turns < self.max_turns:
                remaining = self.response_length - len(response_mask)
                if remaining <= 0:
                    break
                prompt_token_lengths.append(len(prompt_ids))
                turn_sampling = {
                    **sampling_params,
                    "max_tokens": min(self.max_tokens_per_turn, remaining),
                }
                request_id = uuid4().hex
                with simple_timer("generate_sequences", metrics):
                    generated: TokenOutput = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=prompt_ids,
                        sampling_params=turn_sampling,
                    )
                assistant_turns += 1
                (
                    merge_result,
                    response_mask,
                    response_logprobs,
                ) = await self.ct_merge_assistant_token(
                    prompt_ids,
                    generated.token_ids,
                    response_mask,
                    response_logprobs,
                    assistant_logprobs=(
                        generated.log_probs if generated.log_probs else None
                    ),
                )
                prompt_ids = merge_result.token_ids

                content, tool_calls = await self.tool_parser.extract_tool_calls(
                    generated.token_ids,
                    tools,
                )
                messages.append(_assistant_message(content, tool_calls))
                for call in tool_calls:
                    tool_call_signatures.append(
                        (
                            call.name,
                            json.dumps(
                                call.arguments,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        )
                    )
                action, multiple_calls = _action_payload(content, tool_calls)
                if multiple_calls or not action:
                    invalid_tool_calls += 1
                    if not action:
                        break

                previous_messages = list(messages)
                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = await asyncio.wait_for(
                    asyncio.to_thread(env.step, action),
                    timeout=self.env_timeout,
                )
                del observation
                final_reward = float(reward)
                raw_observation = getattr(env._agent, "observation", None)
                for item in raw_observation or []:
                    for message in message_items(item):
                        if (
                            isinstance(message, ToolMessage)
                            and message.id not in seen_tool_message_ids
                        ):
                            seen_tool_message_ids.add(message.id)
                            if message.error:
                                invalid_tool_calls += 1

                if terminated or truncated:
                    simulation = json.loads(info.get("simulation_run") or "{}")
                    termination_reason = simulation.get("termination_reason")
                    reward_info = json.loads(info.get("reward_info") or "{}")
                    break
                new_context = new_context_messages(
                    message_observation,
                    raw_observation,
                )
                message_observation = list(raw_observation or [])
                if new_context:
                    messages.extend(new_context)
                    with simple_timer("tool_calls", metrics):
                        (
                            merge_result,
                            response_mask,
                            response_logprobs,
                        ) = await self.ct_merge_context_msg(
                            previous_messages,
                            messages,
                            prompt_ids,
                            response_mask,
                            response_logprobs,
                            tools=tools,
                        )
                    prompt_ids = merge_result.token_ids
                if len(response_mask) >= self.response_length:
                    break
        finally:
            close = getattr(env, "close", None)
            if close is not None:
                close()

        client_end = self.deepseek_client.snapshot()
        for metric in (
            "attempts",
            "successful_calls",
            "retries",
            "rate_limit_events",
            "timeout_events",
            "failures",
        ):
            metrics[f"deepseek/{metric}_delta"] = (
                client_end[metric] - client_start[metric]
            )

        strict_success = final_reward >= 1.0 - 1e-9
        repeated_actions = max(
            0,
            _maximum_consecutive_repeat(tool_call_signatures) - 1,
        )
        breakdown = compute_agent_reward(
            reward_info=reward_info,
            termination_reason=termination_reason,
            invalid_tool_calls=invalid_tool_calls,
            repeated_actions=repeated_actions,
            strict_success=strict_success,
        )
        metrics["context/prompt_tokens_initial"] = (
            prompt_token_lengths[0] if prompt_token_lengths else 0
        )
        metrics["context/prompt_tokens_final"] = (
            prompt_token_lengths[-1] if prompt_token_lengths else 0
        )
        metrics["context/prompt_tokens_max"] = (
            max(prompt_token_lengths) if prompt_token_lengths else 0
        )
        metrics["context/assistant_turns"] = assistant_turns
        metrics["context/tool_calls"] = len(tool_call_signatures)
        response_ids = prompt_ids[-len(response_mask) :] if response_mask else []
        prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            reward_score=breakdown.total,
            num_turns=assistant_turns,
            metrics=metrics,
            extra_fields={
                "strict_success": strict_success,
                "termination_reason": termination_reason,
                "task_id": task_id,
                "turn_scores": [],
                "tool_rewards": [],
                "premature_exit": breakdown.premature_exit,
            },
        )
        append_context_audit(
            {
                "task_id": task_id,
                "seed": seed,
                "assistant_turns": assistant_turns,
                "tool_calls": len(tool_call_signatures),
                "invalid_tool_calls": invalid_tool_calls,
                "prompt_token_lengths": prompt_token_lengths,
                "prompt_token_deltas": [
                    current - previous
                    for previous, current in zip(
                        prompt_token_lengths,
                        prompt_token_lengths[1:],
                        strict=False,
                    )
                ],
                "strict_success": strict_success,
                "reward_score": breakdown.total,
                "termination_reason": termination_reason,
            }
        )
        return output
