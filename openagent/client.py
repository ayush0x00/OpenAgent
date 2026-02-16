"""Reusable agent client: WebSocket connect, register, then query/result or tool_call/result."""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Awaitable, Callable, Literal

import websockets
from websockets.exceptions import ConnectionClosed

from protocol import (
    Error,
    Ping,
    Pong,
    Query,
    QueryResult,
    Registered,
    ToolCall,
    ToolResult,
    ToolSchema,
    parse_message,
    message_to_json,
)


class AgentClient:
    """Single connection to master. Use as query agent (send queries) or action agent (handle tool_call)."""

    def __init__(
        self,
        master_url: str,
        agent_id: str,
        agent_type: Literal["query", "action"],
        *,
        metadata: dict[str, Any] | None = None,
        tools: list[ToolSchema] | None = None,
        tool_handler: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
    ):
        self.master_url = master_url
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.metadata = metadata or {}
        self.tools = tools or []
        self.tool_handler = tool_handler
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._registered = asyncio.Event()
        self._registration_error: str | None = None
        self._pending_queries: dict[str, asyncio.Future[QueryResult | Error]] = {}
        self._recv_task: asyncio.Task[None] | None = None

    async def connect(self, *, connect_timeout: float = 10.0) -> None:
        self._ws = await asyncio.wait_for(
            websockets.connect(self.master_url, close_timeout=2),
            timeout=connect_timeout,
        )
        self._registered.clear()
        self._registration_error = None
        self._recv_task = asyncio.create_task(self._recv_loop())
        reg_id = str(uuid.uuid4())
        msg = {
            "type": "register",
            "id": reg_id,
            "agent_type": self.agent_type,
            "agent_id": self.agent_id,
            "metadata": self.metadata,
            "ts": None,
        }
        if self.tools:
            msg["tools"] = [t.model_dump() for t in self.tools]
        await self._ws.send(json.dumps(msg))
        await asyncio.wait_for(self._registered.wait(), timeout=10.0)
        if self._registration_error:
            raise RuntimeError(self._registration_error)
        print("Registered with master.")

    async def close(self) -> None:
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def _recv_loop(self) -> None:
        if not self._ws:
            return
        try:
            while True:
                raw = await self._ws.recv()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                msg = parse_message(data)
                if msg is None:
                    continue
                if isinstance(msg, Registered):
                    self._registered.set()
                elif isinstance(msg, Error):
                    if not self._registered.is_set():
                        self._registration_error = msg.message
                        self._registered.set()
                    else:
                        fid = getattr(msg, "id", None)
                        if fid and fid in self._pending_queries:
                            self._pending_queries[fid].set_result(msg)
                elif isinstance(msg, QueryResult):
                    if msg.id in self._pending_queries:
                        self._pending_queries[msg.id].set_result(msg)
                elif isinstance(msg, ToolCall) and self.tool_handler:
                    asyncio.create_task(self._handle_tool_call(msg))
                elif isinstance(msg, Pong):
                    pass
        except ConnectionClosed:
            pass
        finally:
            for f in self._pending_queries.values():
                if not f.done():
                    f.cancel()

    async def _handle_tool_call(self, call: ToolCall) -> None:
        try:
            result = await self.tool_handler(call.tool_name, call.arguments)
            out = ToolResult(
                id=str(uuid.uuid4()),
                call_id=call.id,
                success=True,
                result=result if isinstance(result, (str, dict)) else str(result),
            )
        except Exception as e:
            out = ToolResult(
                id=str(uuid.uuid4()),
                call_id=call.id,
                success=False,
                error=str(e),
            )
        await self._ws.send(message_to_json(out))

    async def query(self, query: str, *, query_id: str | None = None) -> QueryResult | Error:
        """Send a query and wait for query_result or error. Only for query agents."""
        qid = query_id or str(uuid.uuid4())
        fut: asyncio.Future[QueryResult | Error] = asyncio.get_event_loop().create_future()
        self._pending_queries[qid] = fut
        try:
            await self._ws.send(message_to_json(Query(id=qid, query=query)))
            return await asyncio.wait_for(fut, timeout=60.0)
        finally:
            self._pending_queries.pop(qid, None)

    def start_recv_loop(self) -> asyncio.Task[None]:
        """Return the recv task (already started in connect()). For query agents."""
        assert self._recv_task is not None, "Call connect() first"
        return self._recv_task

    async def run_action_agent(self) -> None:
        """Connect and run recv loop forever (handles tool_call). Use for action agents."""
        await self.connect()
        try:
            if self._recv_task:
                await self._recv_task
        finally:
            await self.close()


# Type alias for agent_type literal
