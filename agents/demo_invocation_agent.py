#!/usr/bin/env python3
"""Demo action agent: HTTP server only. Registers invocation URL with master once (Redis). No WebSocket."""
import asyncio
import os
import sys
import threading
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx
import uvicorn
from fastapi import FastAPI, Request

from openagent import ToolSchema, register_invocation_agent

# Invocation server config
INVOCATION_PORT = int(os.environ.get("INVOCATION_PORT", "9000"))
INVOCATION_HOST = os.environ.get("INVOCATION_HOST", "127.0.0.1")
INVOCATION_BASE_URL = os.environ.get(
    "INVOCATION_BASE_URL",
    f"http://{INVOCATION_HOST}:{INVOCATION_PORT}",
)

TOOLS = [
    ToolSchema(
        name="echo",
        description="Echo back a message. Use when the user wants to repeat or echo something.",
        parameters={
            "type": "object",
            "properties": {"message": {"type": "string", "description": "Message to echo back"}},
            "required": ["message"],
        },
        endpoint="/run",
    ),
    ToolSchema(
        name="get_time",
        description="Get the current time. Use when the user asks for the time.",
        parameters={
            "type": "object",
            "properties": {"timezone": {"type": "string", "description": "Timezone, e.g. UTC or America/New_York"}},
            "required": [],
        },
        endpoint="/get_time",
    ),
]


async def handle_tool(tool_name: str, arguments: dict) -> str | dict:
    print(f"[demo-invocation-agent] tool_call: {tool_name}({arguments})")
    if tool_name == "echo":
        msg = arguments.get("message", "")
        return {"echo": msg}
    if tool_name == "get_time":
        tz = arguments.get("timezone") or "UTC"
        now = datetime.now(timezone.utc)
        return {"time": now.isoformat(), "timezone": tz}
    raise ValueError(f"Unknown tool: {tool_name}")


app = FastAPI(title="Demo Invocation Agent")


@app.get("/health")
def health():
    """Health check for master. Master calls GET {invocation_base_url}/health to see if agent is UP."""
    return {"status": "ok"}


async def _run_impl(request: Request):
    body = await request.json()
    call_id = body.get("call_id")
    tool_name = body.get("tool_name")
    arguments = body.get("arguments", {})
    callback_url = body.get("callback_url")
    if not call_id or not tool_name or not callback_url:
        return {"ok": False, "error": "call_id, tool_name, callback_url required"}
    try:
        result = await handle_tool(tool_name, arguments)
        payload = {"call_id": call_id, "success": True, "result": result}
    except Exception as e:
        payload = {"call_id": call_id, "success": False, "error": str(e)}
    async with httpx.AsyncClient() as client:
        await client.post(callback_url, json=payload, timeout=10.0)
    return {"ok": True}


@app.post("/run")
async def run(request: Request):
    return await _run_impl(request)


@app.post("/get_time")
async def get_time(request: Request):
    return await _run_impl(request)


def _run_server():
    uvicorn.run(app, host=INVOCATION_HOST, port=INVOCATION_PORT, log_level="warning")


async def main():
    server_thread = threading.Thread(target=_run_server, daemon=True)
    server_thread.start()
    await asyncio.sleep(0.5)  # let server bind
    print(f"Invocation base: {INVOCATION_BASE_URL} (tools: /run, /get_time)")
    print("Registering with master (HTTP, Redis)...")
    await register_invocation_agent(
        "demo-invocation-agent",
        TOOLS,
        INVOCATION_BASE_URL,
    )
    print("Registered. Server running; master will invoke via URL. Ctrl+C to stop.")
    await asyncio.Event().wait()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down.")
