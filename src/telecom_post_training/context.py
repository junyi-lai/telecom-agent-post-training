from __future__ import annotations

import hashlib
import json
from pathlib import Path


def build_runtime_context() -> dict:
    """Build the exact text-only system prompt and tool schemas used by tau2."""
    from tau2.agent.llm_agent import LLMAgent
    from tau2.domains.telecom.environment import get_environment

    environment = get_environment(solo_mode=False, policy_type="manual")
    tools = environment.get_tools()
    agent = LLMAgent(
        tools=tools,
        domain_policy=environment.policy,
        llm="unused",
        llm_args={"temperature": 0.0},
    )
    tool_schemas = [tool.openai_schema for tool in tools]
    canonical = json.dumps(
        {"system_prompt": agent.system_prompt, "tools": tool_schemas},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "schema_version": 1,
        "domain": "telecom",
        "system_prompt": agent.system_prompt,
        "tools": tool_schemas,
        "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }


def write_runtime_context(destination: Path) -> dict:
    context = build_runtime_context()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(context, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return context
