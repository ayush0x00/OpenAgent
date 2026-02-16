"""Orchestrator: GPT-4o-mini decides answer_directly or call_tool."""
from __future__ import annotations

import json
import re
from typing import Any

from openai import AsyncOpenAI

SYSTEM = """You are an orchestrator. Given a user query and a list of available agents with their tools (each tool has name, description, parameters), you must respond with exactly one JSON object, no other text.

If the query can be answered by calling one of the tools, respond with:
{"action": "call_tool", "agent_id": "<agent_id>", "tool_name": "<tool name>", "arguments": {<tool arguments as key-value pairs>}}

If the query cannot be answered by any tool (e.g. greeting, or no relevant tool), respond with:
{"action": "answer_directly", "text": "<your short answer>"}

Use only agent_id and tool_name from the provided list. Arguments must match the tool's parameters. Respond with valid JSON only."""


async def decide(
    openai_client: AsyncOpenAI,
    model: str,
    query: str,
    agents_snapshot: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Returns either {"action": "answer_directly", "text": "..."} or
    {"action": "call_tool", "agent_id": "...", "tool_name": "...", "arguments": {...}}.
    Raises or returns error dict on parse failure.
    """
    tools_json = json.dumps(agents_snapshot, indent=2)
    user = f"Available agents and tools:\n{tools_json}\n\nUser query: {query}"

    resp = await openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    content = (resp.choices[0].message.content or "").strip()
    # Extract JSON if wrapped in markdown
    m = re.search(r"\{[\s\S]*\}", content)
    if not m:
        return {"action": "answer_directly", "text": f"I couldn't parse a response. Raw: {content[:200]}"}
    try:
        out = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"action": "answer_directly", "text": f"Invalid JSON from model: {content[:200]}"}
    if out.get("action") == "answer_directly":
        return {"action": "answer_directly", "text": out.get("text", "") or "No answer."}
    if out.get("action") == "call_tool":
        return {
            "action": "call_tool",
            "agent_id": out.get("agent_id", ""),
            "tool_name": out.get("tool_name", ""),
            "arguments": out.get("arguments") or {},
        }
    return {"action": "answer_directly", "text": f"Unknown action: {out}"}
