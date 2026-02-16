#!/usr/bin/env python3
"""Example query agent: sends a few queries and prints results."""
import asyncio
import os
import sys

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from protocol import QueryResult
from openagent import AgentClient


async def main():
    url = os.environ.get("MASTER_WS", "ws://127.0.0.1:8000/ws")
    client = AgentClient(
        master_url=url,
        agent_id="demo-query-agent",
        agent_type="query",
    )
    print("Connecting to master...")
    await client.connect()
    recv_task = client.start_recv_loop()

    try:
        for q in ["What's the weather in NYC?", "Hello!", "Weather in London?"]:
            print(f"\nQuery: {q}")
            res = await client.query(q)
            if isinstance(res, QueryResult):
                if res.error:
                    print(f"  Error: {res.error}")
                else:
                    print(f"  Result: {res.result}")
            else:
                print(f"  Error: {getattr(res, 'message', res)}")
    finally:
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass
        await client.close()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
