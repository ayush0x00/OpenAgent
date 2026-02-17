#!/usr/bin/env python3
"""Example query agent: sends a few queries and prints results."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from protocol import QueryResult
from openagent import connect_master


async def main():
    async with connect_master(agent_id="demo-query-agent") as client:
        for q in ["What's the weather in NYC?", "Hello!", "Weather in London?","I need to echo back 'HELLLOOO WORLDDD !!!'"]:
            print(f"\nQuery: {q}")
            res = await client.query(q)
            if isinstance(res, QueryResult):
                if res.error:
                    print(f"  Error: {res.error}")
                else:
                    print(f"  Result: {res.result}")
            else:
                print(f"  Error: {getattr(res, 'message', res)}")
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
