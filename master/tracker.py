"""Tracks all agents (connected + disconnected) for listing and terminal display."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class AgentEntry:
    agent_id: str
    agent_type: str  # "query" | "action"
    metadata: dict[str, Any]
    tools: list[dict[str, Any]]  # [{name, description, parameters}]
    connected: bool
    connection_type: str  # "in_memory" | "invocation"
    invocation_url: str | None = None
    connected_at: str | None = None
    disconnected_at: str | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentTracker:
    def __init__(self) -> None:
        self._agents: dict[str, AgentEntry] = {}

    def on_connect(
        self,
        agent_id: str,
        agent_type: str,
        metadata: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        *,
        invocation_url: str | None = None,
    ) -> None:
        connection_type = "invocation" if invocation_url else "in_memory"
        self._agents[agent_id] = AgentEntry(
            agent_id=agent_id,
            agent_type=agent_type,
            metadata=metadata or {},
            tools=tools or [],
            connected=True,
            connection_type=connection_type,
            invocation_url=invocation_url,
            connected_at=_now(),
            disconnected_at=None,
        )

    def on_disconnect(self, agent_id: str) -> None:
        if agent_id in self._agents:
            self._agents[agent_id].connected = False
            self._agents[agent_id].disconnected_at = _now()

    def list_agents(self) -> list[dict[str, Any]]:
        """For GET /agents: list of agents with status, tools, metadata, connection_type, invocation_url."""
        return [
            {
                "agent_id": e.agent_id,
                "agent_type": e.agent_type,
                "metadata": e.metadata,
                "tools": e.tools,
                "connected": e.connected,
                "connection_type": e.connection_type,
                "invocation_url": e.invocation_url,
                "connected_at": e.connected_at,
                "disconnected_at": e.disconnected_at,
            }
            for e in self._agents.values()
        ]

    def print_status(self) -> None:
        """Print agent list to terminal with green (live) / red (dead) dots."""
        GREEN = "\033[92m"
        RED = "\033[91m"
        RESET = "\033[0m"
        entries = list(self._agents.values())
        if not entries:
            print("  (no agents)")
            return
        for e in entries:
            dot = f"{GREEN}\u2022{RESET}" if e.connected else f"{RED}\u2022{RESET}"
            tools_str = ", ".join(t["name"] for t in e.tools) if e.tools else "—"
            status = "live" if e.connected else "disconnected"
            how = f"invocation: {e.invocation_url}" if e.connection_type == "invocation" and e.invocation_url else "in-memory"
            print(f"  {dot} {e.agent_id} ({e.agent_type}) — tools: [{tools_str}] — {status} — [{how}]")
