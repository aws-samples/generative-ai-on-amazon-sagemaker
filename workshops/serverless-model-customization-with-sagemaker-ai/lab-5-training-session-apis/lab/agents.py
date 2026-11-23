"""Strands Agents helpers shared by the use cases.

The use case modules (for example usecases/scheduling_hard.py) build their agents with these functions. make_model
selects the model: the policy behind the rollout gateway during training, or an Amazon Bedrock model for
comparison. make_agent and run create and run an agent. transcript, count_turns, count_tool_calls and
print_transcript summarize a finished Strands conversation for metrics and printing.
"""

from __future__ import annotations

import json
from typing import Any

from strands import Agent
from strands.agent.conversation_manager import NullConversationManager


def make_model(model_config: dict):
    """Return the Strands model for model_config: the policy behind the rollout gateway, or an Amazon Bedrock model.

    Args:
        model_config: Dict with one of two sets of keys.
            bedrock_model_id, temperature (optional, default 0.0): an Amazon Bedrock model (strands BedrockModel).
            base_url, api_key (optional, default "EMPTY"), model_id (optional, default "policy"), sampling_params
            (optional dict of request parameters): an OpenAI-compatible chat API, which during training is the
            rollout gateway (strands OpenAIModel).

    Returns:
        A strands BedrockModel or OpenAIModel. The model classes are imported only when needed. Creating the model
        sends no request.
    """
    if model_config.get("bedrock_model_id"):
        from strands.models import BedrockModel

        return BedrockModel(model_id=model_config["bedrock_model_id"], temperature=model_config.get("temperature", 0.0))
    from strands.models.openai import OpenAIModel

    return OpenAIModel(
        # max_retries=0: the gateway answers 429 when a rollout exceeds its turn cap.
        client_args={"api_key": model_config.get("api_key") or "EMPTY", "base_url": model_config["base_url"],
                     "max_retries": 0, "timeout": 600},
        model_id=model_config.get("model_id", "policy"),
        params=model_config.get("sampling_params", {}),
    )


def make_agent(model_config: dict, tools: list, system_prompt: str) -> Agent:
    """Return a Strands agent with the model from make_model(model_config), the given tools and system prompt.

    Args:
        model_config: Model settings passed to make_model.
        tools: List of Strands tools (functions decorated with @tool).
        system_prompt: System prompt of the agent.

    Returns:
        A strands Agent. It keeps the full conversation (NullConversationManager, no trimming of old turns) and
        prints nothing while it runs (callback_handler=None).
    """
    return Agent(
        model=make_model(model_config),
        tools=tools,
        system_prompt=system_prompt,
        conversation_manager=NullConversationManager(),
        callback_handler=None,
    )


def run(agent: Agent, prompt: str) -> tuple[str, str | None]:
    """Send prompt to the agent and run it until the model answers without a tool call.

    Args:
        agent: A Strands agent, for example from make_agent.
        prompt: The user message.

    Returns:
        A tuple (final text, error). final text is the text of the last assistant message, or "" on error. error
        is None on success, or "ExceptionType: message" cut to 300 characters. Exceptions (for example a turn or
        context limit) end the episode instead of being raised.

    Side effects: the agent calls its model and tools, and agent.messages holds the conversation afterwards.
    """
    try:
        result = agent(prompt)
        content = result.message.get("content") or []
        return "".join(b["text"] for b in content if "text" in b), None
    except Exception as exc:  # noqa: BLE001
        return "", f"{type(exc).__name__}: {exc}"[:300]


def transcript(messages: list[dict], max_chars: int = 400) -> list[dict]:
    """Return a compact view of a Strands conversation: text, tool calls and tool results.

    Args:
        messages: Strands conversation (agent.messages), a list of message dicts with "role" and "content".
        max_chars: Maximum number of characters kept for each text, tool input and tool result.

    Returns:
        A list of dicts in conversation order. Text blocks give {"role", "text"}, tool calls give
        {"role": "call", "tool", "input"} with the input as JSON, and tool results give
        {"role": "result", "status", "text"}.
    """
    out = []
    for m in messages:
        for block in m.get("content") or []:
            if "text" in block and block["text"].strip():
                out.append({"role": m["role"], "text": block["text"][:max_chars]})
            elif "toolUse" in block:
                tu = block["toolUse"]
                out.append({"role": "call", "tool": tu.get("name"), "input": json.dumps(tu.get("input"))[:max_chars]})
            elif "toolResult" in block:
                tr = block["toolResult"]
                txt = " ".join(c.get("text", "") for c in tr.get("content", []) if isinstance(c, dict))
                out.append({"role": "result", "status": tr.get("status"), "text": txt[:max_chars]})
    return out


def count_turns(messages: list[dict]) -> int:
    """Return the number of assistant messages in messages, a Strands conversation.

    Each assistant message comes from one model call, so this is the number of model calls (turns).
    """
    return sum(1 for m in messages if m["role"] == "assistant")


def count_tool_calls(messages: list[dict]) -> int:
    """Return the number of tool calls (toolUse blocks) in messages, a Strands conversation."""
    return sum(1 for m in messages for b in m.get("content") or [] if "toolUse" in b)


def print_transcript(t: list[dict[str, Any]]) -> None:
    """Print a transcript from transcript() to standard output.

    Tool calls are printed as ->, tool results as <- with their status, and text with its role. Returns None.
    """
    for e in t:
        if e["role"] == "call":
            print(f"  -> {e['tool']}({e['input']})")
        elif e["role"] == "result":
            print(f"  <- [{e.get('status')}] {e['text']}")
        else:
            print(f"  {e['role']}: {e['text']}")
