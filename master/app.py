"""FastAPI app: WebSocket endpoint, registry, orchestrator integration."""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from openai import AsyncOpenAI

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
    parse_message,
    message_to_json,
)
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


def get_openai_client() -> AsyncOpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return AsyncOpenAI(api_key=key)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # cleanup if any


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
                    registry.register_action(agent_id, reg.metadata or {}, reg.tools or [], send_msg)
                    tracker.on_connect(agent_id, agent_type, reg.metadata or {}, tools_for_tracker)
                    print(f"{GREEN}\u2022{RESET} {agent_id} connected (action)")
                    tracker.print_status()
                else:
                    registry.register_query(agent_id, reg.metadata or {}, send_msg)
                await send_msg(Registered(id=reg.id, agent_id=agent_id))
                continue

            if isinstance(parsed, Query):
                # Run orchestrator and either answer_directly or call_tool
                try:
                    client = get_openai_client()
                    snapshot = registry.action_agents_snapshot()
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
                    action_agent = registry.get_action(aid)
                    if not action_agent:
                        await send_msg(QueryResult(id=parsed.id, error=f"Unknown action agent: {aid}"))
                        continue
                    call_id = str(uuid.uuid4())
                    print(f"[Orchestrator] calling {aid}.{tname}({args})")
                    loop = asyncio.get_event_loop()
                    fut = loop.create_future()
                    pending_tool_results[call_id] = (fut, send_msg, parsed.id)
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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/agents")
def list_agents():
    """List all agents (connected and disconnected) with tools/capabilities and status."""
    return {"agents": tracker.list_agents()}
