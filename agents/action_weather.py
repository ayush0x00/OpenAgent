#!/usr/bin/env python3
"""Example action agent: exposes get_weather tool."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openagent import ToolSchema, run_action_agent

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


if __name__ == "__main__":
    asyncio.run(run_action_agent("weather-agent", TOOLS, handle_tool))
