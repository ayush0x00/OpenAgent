"""In-memory agent registry: action agents with tools, query agents (for correlation)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from protocol import ToolSchema


@dataclass
class RegisteredActionAgent:
    agent_id: str
    metadata: dict[str, Any]
    tools: list[ToolSchema]
    send: Any  # async def (msg: BaseModel) -> None


@dataclass
class RegisteredQueryAgent:
    agent_id: str
    metadata: dict[str, Any]
    send: Any  # async def (msg: BaseModel) -> None


class AgentRegistry:
    def __init__(self) -> None:
        self._action: dict[str, RegisteredActionAgent] = {}
        self._query: dict[str, RegisteredQueryAgent] = {}

    def register_action(self, agent_id: str, metadata: dict, tools: list[ToolSchema], send: Any) -> None:
        self._action[agent_id] = RegisteredActionAgent(agent_id=agent_id, metadata=metadata, tools=tools, send=send)

    def register_query(self, agent_id: str, metadata: dict, send: Any) -> None:
        self._query[agent_id] = RegisteredQueryAgent(agent_id=agent_id, metadata=metadata, send=send)

    def unregister(self, agent_id: str) -> None:
        self._action.pop(agent_id, None)
        self._query.pop(agent_id, None)

    def get_action(self, agent_id: str) -> RegisteredActionAgent | None:
        return self._action.get(agent_id)

    def get_query(self, agent_id: str) -> RegisteredQueryAgent | None:
        return self._query.get(agent_id)

    def action_agents_snapshot(self) -> list[dict[str, Any]]:
        """Snapshot for orchestrator: list of {agent_id, metadata, tools: [{name, description, parameters}]}."""
        return [
            {
                "agent_id": a.agent_id,
                "metadata": a.metadata,
                "tools": [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in a.tools],
            }
            for a in self._action.values()
        ]
