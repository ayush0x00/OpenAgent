"""Wire protocol: all message types and fields. Canonical spec."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Union

from pydantic import BaseModel, Field


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- Tool schema (for action agent registration) ---


class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)  # JSON Schema object
    endpoint: str | None = None  # path for this tool, e.g. "/run" or "/get_time"; default "/run"


# --- Agent -> Master ---


class Register(BaseModel):
    type: Literal["register"] = "register"
    id: str
    agent_type: Literal["query", "action"]
    agent_id: str
    metadata: dict[str, Any] | None = None
    tools: list[ToolSchema] | None = None  # required for action
    invocation_url: str | None = None  # optional; when set, master invokes via HTTP
    ts: str | None = None


class Query(BaseModel):
    type: Literal["query"] = "query"
    id: str
    query: str
    ts: str | None = None


class ToolResult(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    id: str
    call_id: str
    success: bool
    result: str | dict[str, Any] | None = None
    error: str | None = None
    ts: str | None = None


class Ping(BaseModel):
    type: Literal["ping"] = "ping"
    id: str
    ts: str | None = None


# --- Master -> Agent ---


class Registered(BaseModel):
    type: Literal["registered"] = "registered"
    id: str
    agent_id: str
    ts: str | None = None


class QueryResult(BaseModel):
    type: Literal["query_result"] = "query_result"
    id: str
    result: str | dict[str, Any] | None = None
    error: str | None = None
    ts: str | None = None


class ToolCall(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    ts: str | None = None


class Pong(BaseModel):
    type: Literal["pong"] = "pong"
    id: str
    ts: str | None = None


class Error(BaseModel):
    type: Literal["error"] = "error"
    id: str
    code: str
    message: str
    ts: str | None = None


# --- Union types for parsing ---

AgentToMaster = Register | Query | ToolResult | Ping
MasterToAgent = Registered | QueryResult | ToolCall | Pong | Error


def parse_message(raw: dict[str, Any]) -> AgentToMaster | MasterToAgent | None:
    """Parse JSON dict into typed message. Returns None if unknown type."""
    t = raw.get("type")
    if not t:
        return None
    models: dict[str, type[BaseModel]] = {
        "register": Register,
        "query": Query,
        "tool_result": ToolResult,
        "ping": Ping,
        "registered": Registered,
        "query_result": QueryResult,
        "tool_call": ToolCall,
        "pong": Pong,
        "error": Error,
    }
    cls = models.get(t)
    if not cls:
        return None
    try:
        return cls.model_validate(raw)
    except Exception:
        return None


def message_to_json(msg: BaseModel) -> str:
    """Serialize message to JSON string."""
    return msg.model_dump_json(exclude_none=True)
