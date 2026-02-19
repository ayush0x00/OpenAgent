#!/usr/bin/env python3
"""
Single load-test server: 1 of 10 FastAPI apps with complex agents (math, weather, text, time, data, code, finance, stats, search, workflow).
Usage: python -m experiments.load_server <server_index>
  server_index in 0..9 → port 9000 + index, uses experiments.complex_apps.
"""
from __future__ import annotations

import asyncio
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx
import uvicorn
from fastapi import FastAPI, Request

import config
from openagent import ToolSchema, register_invocation_agent
from openagent.client import AgentClient

from experiments.complex_apps import get_app_tools_and_handler
from experiments.complex_apps.apps import THEMES

BASE_PORT = 9000
NUM_COMPLEX_APPS = 10
NUM_PROCESS_AGENTS = 5  # per server


def make_process_agent_tools(server_index: int, agent_index: int) -> list[ToolSchema]:
    prefix = f"proc_s{server_index}_a{agent_index}"
    return [
        ToolSchema(name=f"{prefix}_ping", description=f"Ping from process agent {agent_index} on server {server_index}.", parameters={"type": "object", "properties": {}, "required": []}),
        ToolSchema(name=f"{prefix}_uppercase", description=f"Uppercase a string. Process agent {agent_index} server {server_index}.", parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}),
    ]


async def process_agent_tool_handler(server_index: int, agent_index: int, tool_name: str, arguments: dict) -> dict:
    if "ping" in tool_name:
        return {"pong": True, "server": server_index, "agent": agent_index}
    if "uppercase" in tool_name:
        return {"result": str(arguments.get("text", "")).upper()}
    return {"ok": True}


async def run_process_agent(server_index: int, agent_index: int) -> None:
    agent_id = f"process-{server_index}-{agent_index}"
    tools = make_process_agent_tools(server_index, agent_index)

    async def handler(tool_name: str, arguments: dict):
        return await process_agent_tool_handler(server_index, agent_index, tool_name, arguments)

    client = AgentClient(master_url=config.MASTER_WS, agent_id=agent_id, agent_type="action", tools=tools, tool_handler=handler, metadata={"server": server_index, "process_agent": agent_index})
    try:
        await client.run_action_agent()
    except asyncio.CancelledError:
        pass
    finally:
        await client.close()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m experiments.load_server <server_index>", file=sys.stderr)
        sys.exit(1)
    server_index = int(sys.argv[1])
    if not 0 <= server_index < NUM_COMPLEX_APPS:
        print(f"server_index must be 0..{NUM_COMPLEX_APPS - 1}", file=sys.stderr)
        sys.exit(1)

    port = BASE_PORT + server_index
    agent_id = f"complex-app-{server_index}"
    tools, _ = get_app_tools_and_handler(server_index)
    host = getattr(config, "INVOCATION_HOST", "127.0.0.1")
    base_url = f"http://{host}:{port}"

    theme = THEMES[server_index] if server_index < len(THEMES) else "app"
    app = FastAPI(title=f"Complex App {server_index} ({theme})")

    @app.get("/health")
    def health():
        return {"status": "ok", "server": server_index}

    tools, handler = get_app_tools_and_handler(server_index)

    async def run_impl(request: Request):
        body = await request.json()
        call_id = body.get("call_id")
        tool_name = body.get("tool_name")
        arguments = body.get("arguments", {})
        callback_url = body.get("callback_url")
        if not call_id or not tool_name or not callback_url:
            return {"ok": False, "error": "call_id, tool_name, callback_url required"}
        try:
            result = handler(server_index, tool_name, arguments)
            payload = {"call_id": call_id, "success": True, "result": result}
        except Exception as e:
            payload = {"call_id": call_id, "success": False, "error": str(e)}
        async with httpx.AsyncClient() as client:
            await client.post(callback_url, json=payload, timeout=10.0)
        return {"ok": True}

    for t in tools:
        endpoint = t.endpoint or f"/{t.name}"
        app.add_api_route(endpoint, run_impl, methods=["POST"])

    def run_uvicorn():
        uvicorn.run(app, host=host, port=port, log_level="warning")

    thread = threading.Thread(target=run_uvicorn, daemon=True)
    thread.start()
    time.sleep(1.0)

    async def main_async():
        if os.environ.get("OPENAGENT_SKIP_HTTP_REGISTER") != "1":
            await register_invocation_agent(agent_id, tools, base_url)
            print(f"[load_server {server_index}] Registered HTTP agent {agent_id} ({len(tools)} tools) on {base_url}")
        else:
            print(f"[load_server {server_index}] HTTP agent {agent_id} on {base_url} (orchestrator will register)")
        process_tasks = [asyncio.create_task(run_process_agent(server_index, j)) for j in range(NUM_PROCESS_AGENTS)]
        print(f"[load_server {server_index}] Started {NUM_PROCESS_AGENTS} process agents")
        try:
            await asyncio.gather(*process_tasks)
        except asyncio.CancelledError:
            for t in process_tasks:
                t.cancel()
            await asyncio.gather(*process_tasks, return_exceptions=True)

    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print(f"[load_server {server_index}] Shutting down")


if __name__ == "__main__":
    main()
