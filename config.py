"""Central configuration. Loads from environment with sensible defaults."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# --- Master ---
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
# Redis agent key TTL (seconds). Unused agents expire; used agents get TTL refreshed. Default 1 day.
REDIS_AGENT_TTL_SECONDS = int(float(os.environ.get("REDIS_AGENT_TTL_SECONDS", "86400")))
MASTER_BASE_URL = os.environ.get("MASTER_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Orchestrator
ORCHESTRATOR_MODEL = os.environ.get("ORCHESTRATOR_MODEL", "gpt-4o-mini")

# Timeouts (seconds)
HEALTH_CHECK_TIMEOUT = float(os.environ.get("HEALTH_CHECK_TIMEOUT", "2.0"))
TOOL_CALL_TIMEOUT = float(os.environ.get("TOOL_CALL_TIMEOUT", "30.0"))
TOOL_INVOKE_HTTP_TIMEOUT = float(os.environ.get("TOOL_INVOKE_HTTP_TIMEOUT", "2.0"))

# --- Agents (master URL when connecting as client) ---
MASTER_WS = os.environ.get("MASTER_WS", "ws://127.0.0.1:8000/ws")

# Agent client timeouts
WS_CONNECT_TIMEOUT = float(os.environ.get("WS_CONNECT_TIMEOUT", "10.0"))
WS_REGISTRATION_TIMEOUT = float(os.environ.get("WS_REGISTRATION_TIMEOUT", "10.0"))
QUERY_RESPONSE_TIMEOUT = float(os.environ.get("QUERY_RESPONSE_TIMEOUT", "60.0"))
HTTP_REGISTER_TIMEOUT = float(os.environ.get("HTTP_REGISTER_TIMEOUT", "10.0"))

# --- Invocation agent (demo / HTTP-registered agents) ---
INVOCATION_PORT = int(os.environ.get("INVOCATION_PORT", "9000"))
INVOCATION_HOST = os.environ.get("INVOCATION_HOST", "127.0.0.1")
INVOCATION_BASE_URL = os.environ.get(
    "INVOCATION_BASE_URL",
    f"http://{INVOCATION_HOST}:{INVOCATION_PORT}",
)
