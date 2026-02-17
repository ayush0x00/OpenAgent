"""FastAPI app: WebSocket endpoint, registry, orchestrator integration, Redis cache, HTTP invocation."""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from redis.asyncio import Redis

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from protocol import (
    Error,
    Ping,
    Pong,
    Query,
    QueryResult,
    Registered,
    Register,
    ToolCall,
    ToolResult,
    ToolSchema,
    parse_message,
    message_to_json,
)
from master.cache import get_agent, get_all_action_agents_snapshot, get_all_cached_agents, save_agent
from master.registry import AgentRegistry
from master.orchestrator import decide
from master.tracker import AgentTracker

# App state
registry = AgentRegistry()
tracker = AgentTracker()
pending_tool_results: dict[str, tuple[asyncio.Future[ToolResult], Any, str]] = {}  # call_id -> (future, query_send, query_id)

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

MASTER_BASE_URL = os.environ.get("MASTER_BASE_URL", "http://127.0.0.1:8000")

# Set in lifespan for use in WS (no request.app in websocket handler)
_redis: Redis | None = None


async def _fetch_cached_agents_with_health() -> list[dict[str, Any]]:
    """Load agents from Redis and run health check on each with invocation_base_url. Returns list of {agent_id, ... status}."""
    if not _redis:
        return []
    agents = await get_all_cached_agents(_redis)
    out = []
    for a in agents:
        base = a.get("invocation_base_url")
        tools = a.get("tools") or []
        entry = {
            "agent_id": a.get("agent_id", "?"),
            "metadata": a.get("metadata") or {},
            "tools": tools,
            "invocation_base_url": base,
        }
        if base:
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    r = await client.get(f"{base.rstrip('/')}/health")
                entry["status"] = "UP" if r.status_code == 200 else "DOWN"
            except Exception:
                entry["status"] = "DOWN"
        else:
            entry["status"] = "in_memory"
        out.append(entry)
    return out


def get_openai_client() -> AsyncOpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return AsyncOpenAI(api_key=key)


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    redis_client: Redis | None = None
    try:
        redis_client = Redis.from_url(redis_url, decode_responses=False)
        await redis_client.ping()
        print(f"[Master] Connected to Redis ({redis_url})")
    except Exception as e:
        print(f"[Master] Redis not available ({e}), using in-memory only")
        redis_client = None
    app.state.redis = redis_client
    global _redis
    _redis = redis_client
    if redis_client:
        agents_with_status = await _fetch_cached_agents_with_health()
        if agents_with_status:
            print("[Master] Known agents (from Redis):")
            for a in agents_with_status:
                base = a.get("invocation_base_url") or "(no base URL, in-memory)"
                tools = a.get("tools") or []
                tool_str = ", ".join(f"{t.get('name')} @ {t.get('endpoint', '/run')}" for t in tools) if tools else "—"
                status = a.get("status", "?")
                if status == "UP":
                    dot = f"{GREEN}\u2022{RESET}"
                elif status == "DOWN":
                    dot = f"{RED}\u2022{RESET}"
                else:
                    dot = "•"
                print(f"  {dot} {a.get('agent_id', '?')} — {base} — {status} — tools: [{tool_str}]")
        else:
            print("[Master] No agents in Redis yet.")
    yield
    _redis = None
    if redis_client:
        await redis_client.aclose()


app = FastAPI(title="OpenAgent Master", lifespan=lifespan)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    agent_id: str | None = None
    agent_type: str | None = None

    async def send_msg(msg: Any):
        await ws.send_text(message_to_json(msg))

    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await send_msg(Error(id=str(uuid.uuid4()), code="invalid_json", message="Invalid JSON"))
                continue
            parsed = parse_message(data)
            if parsed is None:
                await send_msg(Error(id=data.get("id", ""), code="unknown_type", message="Unknown message type"))
                continue

            if agent_id is None:
                if not isinstance(parsed, Register):
                    await send_msg(Error(id=data.get("id", ""), code="register_first", message="Send register first"))
                    continue
                reg = parsed
                if reg.agent_type == "action" and (not reg.tools or len(reg.tools) == 0):
                    await send_msg(Error(id=reg.id, code="tools_required", message="Action agents must provide tools"))
                    continue
                agent_id = reg.agent_id
                agent_type = reg.agent_type
                tools_for_tracker = [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in (reg.tools or [])]
                if agent_type == "action":
                    invocation_url = getattr(reg, "invocation_url", None) or None
                    registry.register_action(agent_id, reg.metadata or {}, reg.tools or [], send_msg)
                    # Only store in Redis when agent has an invocation URL (survives restart). In-memory-only agents are not cached.
                    if _redis and invocation_url:
                        await save_agent(
                            _redis,
                            agent_id,
                            reg.tools or [],
                            reg.metadata or {},
                            invocation_url=invocation_url,
                        )
                    tracker.on_connect(
                        agent_id,
                        agent_type,
                        reg.metadata or {},
                        tools_for_tracker,
                        invocation_url=invocation_url,
                    )
                    how = f"invocation: {invocation_url}" if invocation_url else "in-memory"
                    print(f"{GREEN}\u2022{RESET} {agent_id} connected (action) — [{how}]")
                    tracker.print_status()
                else:
                    registry.register_query(agent_id, reg.metadata or {}, send_msg)
                await send_msg(Registered(id=reg.id, agent_id=agent_id))
                continue

            if isinstance(parsed, Query):
                # Run orchestrator and either answer_directly or call_tool (snapshot from Redis, fallback to registry)
                try:
                    client = get_openai_client()
                    snapshot = await get_all_action_agents_snapshot(_redis) if _redis else []
                    registry_snapshot = registry.action_agents_snapshot()
                    seen = {a["agent_id"] for a in snapshot}
                    for a in registry_snapshot:
                        if a["agent_id"] not in seen:
                            snapshot.append(a)
                            seen.add(a["agent_id"])
                    decision = await decide(client, "gpt-4o-mini", parsed.query, snapshot)
                except Exception as e:
                    await send_msg(QueryResult(id=parsed.id, error=str(e)))
                    continue
                if decision.get("action") == "answer_directly":
                    await send_msg(QueryResult(id=parsed.id, result=decision.get("text", "")))
                    continue
                if decision.get("action") == "call_tool":
                    aid = decision.get("agent_id", "")
                    tname = decision.get("tool_name", "")
                    args = decision.get("arguments") or {}
                    call_id = str(uuid.uuid4())
                    loop = asyncio.get_event_loop()
                    fut = loop.create_future()
                    pending_tool_results[call_id] = (fut, send_msg, parsed.id)

                    cached = await get_agent(_redis, aid) if _redis else None
                    invocation_url = None
                    if cached:
                        base = cached.get("invocation_base_url")
                        tools_cached = cached.get("tools") or []
                        tool_entry = next((t for t in tools_cached if t.get("name") == tname), None)
                        endpoint = (tool_entry or {}).get("endpoint", "/run")
                        if not endpoint.startswith("/"):
                            endpoint = "/" + endpoint
                        if base:
                            invocation_url = f"{base.rstrip('/')}{endpoint}"
                        else:
                            invocation_url = cached.get("invocation_url")  # legacy

                    if invocation_url:
                        # HTTP invoke: POST to per-tool URL; callback will complete the future
                        callback_url = f"{MASTER_BASE_URL.rstrip('/')}/tool_callback"
                        payload = {"call_id": call_id, "tool_name": tname, "arguments": args, "callback_url": callback_url}
                        print(f"[Orchestrator] calling {aid}.{tname}({args}) via HTTP @ {invocation_url}")
                        try:
                            async with httpx.AsyncClient(timeout=2.0) as http:
                                await http.post(invocation_url, json=payload)
                        except Exception as e:
                            pending_tool_results.pop(call_id, None)
                            await send_msg(QueryResult(id=parsed.id, error=f"Invocation failed: {e}"))
                            continue
                    else:
                        # WS invoke (in-memory registry)
                        action_agent = registry.get_action(aid)
                        if not action_agent:
                            pending_tool_results.pop(call_id, None)
                            await send_msg(QueryResult(id=parsed.id, error=f"Unknown action agent: {aid}"))
                            continue
                        print(f"[Orchestrator] calling {aid}.{tname}({args}) via WS")
                        await action_agent.send(ToolCall(id=call_id, tool_name=tname, arguments=args))

                    try:
                        tool_res: ToolResult = await asyncio.wait_for(fut, timeout=30.0)
                    except asyncio.TimeoutError:
                        pending_tool_results.pop(call_id, None)
                        print(f"[Orchestrator] {aid}.{tname} -> timeout")
                        await send_msg(QueryResult(id=parsed.id, error="Tool call timed out"))
                        continue
                    pending_tool_results.pop(call_id, None)
                    if tool_res.success:
                        print(f"[Orchestrator] {aid}.{tname} -> ok: {tool_res.result}")
                        await send_msg(QueryResult(id=parsed.id, result=tool_res.result))
                    else:
                        print(f"[Orchestrator] {aid}.{tname} -> error: {tool_res.error}")
                        await send_msg(QueryResult(id=parsed.id, error=tool_res.error or "Tool failed"))
                    continue

            if isinstance(parsed, ToolResult):
                entry = pending_tool_results.get(parsed.call_id)
                if entry:
                    fut, _, _ = entry
                    if not fut.done():
                        fut.set_result(parsed)
                continue

            if isinstance(parsed, Ping):
                await send_msg(Pong(id=parsed.id))
                continue

    except WebSocketDisconnect:
        pass
    finally:
        if agent_id:
            registry.unregister(agent_id)
            if agent_type == "action":
                tracker.on_disconnect(agent_id)
                print(f"{RED}\u2022{RESET} {agent_id} disconnected")
                tracker.print_status()


@app.post("/tool_callback")
async def tool_callback(request: Request):
    """Agent invocation endpoint calls this with call_id, success, result/error."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "Invalid JSON"}, status_code=400)
    call_id = body.get("call_id")
    success = body.get("success", False)
    result = body.get("result")
    error = body.get("error")
    if not call_id:
        return JSONResponse({"ok": False, "error": "call_id required"}, status_code=400)
    entry = pending_tool_results.get(call_id)
    if not entry:
        return JSONResponse({"ok": False, "error": "Unknown or expired call_id"}, status_code=404)
    fut, send_msg, query_id = entry
    if not fut.done():
        tool_res = ToolResult(
            id=str(uuid.uuid4()),
            call_id=call_id,
            success=success,
            result=result,
            error=error,
        )
        fut.set_result(tool_res)
    return JSONResponse({"ok": True})


@app.post("/register")
async def register_agent(request: Request):
    """Register an action agent (Redis only). Use invocation_base_url + per-tool endpoint, or invocation_url (legacy)."""
    if not _redis:
        return JSONResponse(
            {"ok": False, "error": "Redis not connected. HTTP registration requires Redis."},
            status_code=503,
        )
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "Invalid JSON"}, status_code=400)
    agent_id = body.get("agent_id")
    invocation_base_url = body.get("invocation_base_url")
    invocation_url = body.get("invocation_url")
    tools_raw = body.get("tools", [])
    if not agent_id:
        return JSONResponse({"ok": False, "error": "agent_id required"}, status_code=400)
    if not invocation_base_url and not invocation_url:
        return JSONResponse(
            {"ok": False, "error": "invocation_base_url or invocation_url required"},
            status_code=400,
        )
    if not tools_raw:
        return JSONResponse(
            {"ok": False, "error": "tools required (list of {name, description, parameters, endpoint?})"},
            status_code=400,
        )
    try:
        tools = [ToolSchema(**t) for t in tools_raw]
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Invalid tools: {e}"}, status_code=400)
    metadata = body.get("metadata") or {}
    await save_agent(
        _redis,
        agent_id,
        tools,
        metadata,
        invocation_base_url=invocation_base_url or None,
        invocation_url=invocation_url if not invocation_base_url else None,
    )
    print(f"[Master] Registered (HTTP) {agent_id} — base: {invocation_base_url or invocation_url}")
    return JSONResponse({"ok": True, "agent_id": agent_id})


@app.get("/refresh")
async def refresh():
    """Re-fetch agents from Redis and run health checks. Returns agents with status (UP/DOWN/in_memory). Prints to terminal."""
    agents_with_status = await _fetch_cached_agents_with_health()
    if agents_with_status:
        print("[Master] Refresh — known agents:")
        for a in agents_with_status:
            base = a.get("invocation_base_url") or "(no base URL, in-memory)"
            tools = a.get("tools") or []
            tool_str = ", ".join(f"{t.get('name')} @ {t.get('endpoint', '/run')}" for t in tools) if tools else "—"
            status = a.get("status", "?")
            if status == "UP":
                dot = f"{GREEN}\u2022{RESET}"
            elif status == "DOWN":
                dot = f"{RED}\u2022{RESET}"
            else:
                dot = "•"
            print(f"  {dot} {a.get('agent_id', '?')} — {base} — {status} — tools: [{tool_str}]")
    else:
        print("[Master] Refresh — no agents in Redis.")
    return {"agents": agents_with_status, "count": len(agents_with_status)}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/agents")
async def list_agents():
    """List all agents: WS tracker (connected/disconnected) + Redis-cached invocation agents."""
    tracker_agents = {e["agent_id"]: e for e in tracker.list_agents()}
    if _redis:
        for cached in await get_all_cached_agents(_redis):
            aid = cached.get("agent_id")
            if aid and aid not in tracker_agents:
                tools = cached.get("tools") or []
                tracker_agents[aid] = {
                    "agent_id": aid,
                    "agent_type": "action",
                    "metadata": cached.get("metadata") or {},
                    "tools": tools,
                    "connected": False,
                    "connection_type": "invocation",
                    "invocation_url": None,
                    "invocation_base_url": cached.get("invocation_base_url"),
                    "connected_at": None,
                    "disconnected_at": None,
                }
    return {"agents": list(tracker_agents.values())}
