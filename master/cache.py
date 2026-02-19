"""Redis cache for agent registrations: invocation_base_url, per-tool endpoint, metadata. Keys have TTL; refresh on use."""
from __future__ import annotations

import json
from typing import Any

from protocol import ToolSchema

# REDIS_URL is in config; cache is used by master which imports config
AGENTS_ACTION_KEY = "agents:action"
AGENT_KEY_PREFIX = "agent:"
DEFAULT_TOOL_ENDPOINT = "/run"


def _agent_ttl_seconds() -> int:
    import config
    return getattr(config, "REDIS_AGENT_TTL_SECONDS", 86400)


def _agent_key(agent_id: str) -> str:
    return f"{AGENT_KEY_PREFIX}{agent_id}"


def _tool_to_dict(t: ToolSchema) -> dict[str, Any]:
    d = {"name": t.name, "description": t.description, "parameters": t.parameters}
    if t.endpoint is not None:
        d["endpoint"] = t.endpoint
    else:
        d["endpoint"] = DEFAULT_TOOL_ENDPOINT
    return d


def _base_and_path_from_url(url: str) -> tuple[str, str]:
    from urllib.parse import urlparse
    p = urlparse(url)
    base = f"{p.scheme}://{p.netloc}"
    path = p.path if p.path else DEFAULT_TOOL_ENDPOINT
    if not path.startswith("/"):
        path = "/" + path
    return base, path


async def save_agent(
    redis_client: Any,
    agent_id: str,
    tools: list[ToolSchema],
    metadata: dict[str, Any],
    *,
    invocation_base_url: str | None = None,
    invocation_url: str | None = None,
) -> None:
    """Store action agent in Redis. Use invocation_base_url + per-tool endpoint, or legacy invocation_url (one URL for all)."""
    base = invocation_base_url
    default_path = DEFAULT_TOOL_ENDPOINT
    if base is None and invocation_url:
        base, default_path = _base_and_path_from_url(invocation_url)
    tools_payload = [_tool_to_dict(t) for t in tools]
    if invocation_url and not invocation_base_url:
        for t in tools_payload:
            if t.get("endpoint") == DEFAULT_TOOL_ENDPOINT:
                t["endpoint"] = default_path
    payload = {
        "agent_id": agent_id,
        "metadata": metadata,
        "tools": tools_payload,
        "invocation_base_url": base or "",
    }
    key = _agent_key(agent_id)
    ttl = _agent_ttl_seconds()
    await redis_client.setex(key, ttl, json.dumps(payload))
    await redis_client.sadd(AGENTS_ACTION_KEY, agent_id)


async def refresh_agent_ttl(redis_client: Any, agent_id: str) -> None:
    """Refresh TTL on agent key (call after successful health check or tool invocation)."""
    key = _agent_key(agent_id)
    await redis_client.expire(key, _agent_ttl_seconds())


async def get_agent(redis_client: Any, agent_id: str) -> dict[str, Any] | None:
    """Load action agent from Redis. Returns None if not found (or expired)."""
    key = _agent_key(agent_id)
    raw = await redis_client.get(key)
    if raw is None:
        return None
    return json.loads(raw)


def _decode_id(aid: Any) -> str:
    return aid.decode() if isinstance(aid, bytes) else aid


async def get_all_cached_agents(redis_client: Any) -> list[dict[str, Any]]:
    """All agents in Redis (full docs). Expired keys are skipped and removed from the set."""
    agent_ids = await redis_client.smembers(AGENTS_ACTION_KEY)
    if not agent_ids:
        return []
    out = []
    for aid in agent_ids:
        agent_id = _decode_id(aid)
        a = await get_agent(redis_client, agent_id)
        if a is not None:
            out.append(a)
        else:
            await redis_client.srem(AGENTS_ACTION_KEY, agent_id)
    return out


async def get_all_action_agents_snapshot(redis_client: Any) -> list[dict[str, Any]]:
    """Snapshot for orchestrator: list of {agent_id, metadata, tools}. Expired keys are skipped and removed from the set."""
    agent_ids = await redis_client.smembers(AGENTS_ACTION_KEY)
    if not agent_ids:
        return []
    out = []
    for aid in agent_ids:
        agent_id = _decode_id(aid)
        a = await get_agent(redis_client, agent_id)
        if a is None:
            await redis_client.srem(AGENTS_ACTION_KEY, agent_id)
            continue
        tools_for_llm = [{"name": t["name"], "description": t["description"], "parameters": t.get("parameters", {})} for t in (a.get("tools") or [])]
        out.append({
            "agent_id": a["agent_id"],
            "metadata": a.get("metadata") or {},
            "tools": tools_for_llm,
        })
    return out


async def delete_agent(redis_client: Any, agent_id: str) -> None:
    """Remove agent from cache (e.g. on disconnect if desired). Optional."""
    await redis_client.delete(_agent_key(agent_id))
    await redis_client.srem(AGENTS_ACTION_KEY, agent_id)
