#!/usr/bin/env python3
"""Example action agent: exposes get_weather tool."""
import asyncio
import os
import sys

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from protocol import ToolSchema
from openagent import AgentClient

TOOLS = [
    ToolSchema(
        name="get_weather",
        description="Get current weather for a city. Use when the user asks about weather.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    ),
]


async def handle_tool(tool_name: str, arguments: dict) -> str | dict:
    print(f"[weather-agent] received tool_call: {tool_name}({arguments})")
    if tool_name == "get_weather":
        city = arguments.get("city", "unknown")
        print(f"[weather-agent] get_weather for city={city}")
        result = {"city": city, "temperature": 72, "unit": "F", "conditions": "sunny"}
        print(f"[weather-agent] returning: {result}")
        return result
    raise ValueError(f"Unknown tool: {tool_name}")


async def main():
    url = os.environ.get("MASTER_WS", "ws://127.0.0.1:8000/ws")
    client = AgentClient(
        master_url=url,
        agent_id="weather-agent",
        agent_type="action",
        tools=TOOLS,
        tool_handler=handle_tool,
    )
    print("Connecting to master as action agent (get_weather)...")
    try:
        await client.run_action_agent()
    except asyncio.TimeoutError:
        print("Connection or registration timed out. Is the master running on MASTER_WS?")
        raise


if __name__ == "__main__":
    asyncio.run(main())
