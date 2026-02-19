#!/usr/bin/env python3
"""
Start 10 load servers (FastAPI complex agents), register them with the orchestrator, run queries, then stop.
Requires: orchestrator (master) running on 8000 with Redis, OPENAI_API_KEY set.

Usage:
  python -m experiments.run_load_test start   # start 10 servers, register with orchestrator
  python -m experiments.run_load_test stop   # kill all load servers
  python -m experiments.run_load_test run <test_name>    # start, register, run, save results, stop
  python -m experiments.run_load_test queries <test_name> # run only, save results (servers already up)
Results are saved under experiments/results/<test_name>_<timestamp>.json
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx
from openagent import connect_master

NUM_SERVERS = 10  # 10 FastAPI apps (complex agents)
BASE_PORT = 9000
MASTER_BASE = os.environ.get("MASTER_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
MASTER_WS = os.environ.get("MASTER_WS", "ws://127.0.0.1:8000/ws")

# PIDs of started server processes (for stop)
_server_pids: list[int] = []

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _sanitize_test_name(name: str) -> str:
    """Safe filename from test name."""
    s = re.sub(r"[^\w\-.]", "_", name.strip()) or "run"
    return s[:200]


def save_results(test_name: str, queries: list[str], stats: dict, metrics: dict) -> str:
    """Write metrics + results to experiments/results/<test_name>_<timestamp>.json. Returns path."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe = _sanitize_test_name(test_name)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"{safe}_{ts}.json")
    payload = {
        "test_name": test_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "num_queries": len(queries),
        "queries": queries,
        "results": [{"status": s, "value": v} for s, v in stats["results"]],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def start_servers() -> None:
    """Spawn 10 load_server processes (they skip HTTP register; we register from here)."""
    global _server_pids
    _server_pids = []
    env = {**os.environ, "OPENAGENT_SKIP_HTTP_REGISTER": "1"}
    for i in range(NUM_SERVERS):
        proc = subprocess.Popen(
            [sys.executable, "-m", "experiments.load_server", str(i)],
            cwd=os.path.join(os.path.dirname(__file__), ".."),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
        )
        _server_pids.append(proc.pid)
    print(f"Started {NUM_SERVERS} servers (PIDs: {_server_pids[:5]}...). Waiting for servers to bind.")
    time.sleep(3)


def _wait_for_health(port: int, timeout_sec: float = 10) -> bool:
    deadline = time.perf_counter() + timeout_sec
    while time.perf_counter() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


def register_all_agents_with_master() -> None:
    """Register all 10 FastAPI agents with the orchestrator (HTTP POST /register), like demo_invocation_agent."""
    from experiments.complex_apps import get_app_tools_and_handler

    host = os.environ.get("INVOCATION_HOST", "127.0.0.1")
    for i in range(NUM_SERVERS):
        port = BASE_PORT + i
        if not _wait_for_health(port):
            print(f"[register] Server {i} (port {port}) not healthy, skipping.")
            continue
        tools, _ = get_app_tools_and_handler(i)
        agent_id = f"complex-app-{i}"
        invocation_base_url = f"http://{host}:{port}"
        payload = {
            "agent_id": agent_id,
            "invocation_base_url": invocation_base_url,
            "tools": [t.model_dump(exclude_none=True) for t in tools],
            "metadata": {"server_index": i},
        }
        try:
            r = httpx.post(f"{MASTER_BASE}/register", json=payload, timeout=10.0)
            r.raise_for_status()
            data = r.json()
            if not data.get("ok"):
                print(f"[register] {agent_id}: {data.get('error', 'failed')}")
            else:
                print(f"[register] {agent_id} OK ({len(tools)} tools) @ {invocation_base_url}")
        except Exception as e:
            print(f"[register] {agent_id}: {e}")


def stop_servers() -> None:
    """Kill all load server processes we started."""
    global _server_pids
    for pid in _server_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    _server_pids = []
    print("Stopped all load servers.")


def kill_all_load_servers_by_port() -> None:
    """Fallback: kill any process listening on BASE_PORT..BASE_PORT+NUM_SERVERS-1."""
    for port in range(BASE_PORT, BASE_PORT + NUM_SERVERS):
        try:
            subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, check=False)
            out = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)
            if out.stdout.strip():
                for pid in out.stdout.strip().split("\n"):
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                    except (ValueError, ProcessLookupError):
                        pass
        except Exception:
            pass
    print("Killed processes on load server ports.")


from experiments.complex_queries import (
    compute_metrics,
    format_metrics_report,
    get_all_queries,
    get_stress_queries,
)

LOAD_TEST_QUERIES = get_all_queries()


async def run_queries(concurrency: int = 5, max_queries: int | None = None, queries: list[str] | None = None) -> dict:
    """Run load-test queries via master. Returns stats."""
    queries = queries or LOAD_TEST_QUERIES[: (max_queries or len(LOAD_TEST_QUERIES))]
    results = []
    errors = 0
    start = time.perf_counter()

    async def run_one(query: str):
        async with connect_master("load-test-query-agent", master_url=MASTER_WS) as client:
            r = await client.query(query)
            return r

    sem = asyncio.Semaphore(concurrency)

    async def run_with_sem(q: str):
        async with sem:
            try:
                r = await run_one(q)
                return ("ok", r)
            except Exception as e:
                return ("error", str(e))

    tasks = [run_with_sem(q) for q in queries]
    out = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.perf_counter() - start

    for i, o in enumerate(out):
        if isinstance(o, BaseException):
            results.append(("exception", str(o)))
            errors += 1
        elif o[0] == "ok":
            res = o[1]
            if getattr(res, "error", None):
                results.append(("error", res.error))
                errors += 1
            else:
                results.append(("ok", getattr(res, "result", res)))
        else:
            results.append(o)
            errors += 1

    return {
        "total": len(queries),
        "ok": len(queries) - errors,
        "errors": errors,
        "elapsed_sec": round(elapsed, 2),
        "qps": round(len(queries) / elapsed, 2) if elapsed > 0 else 0,
        "results": results,
    }


def _pid_file() -> str:
    return os.path.join(os.path.dirname(__file__), ".load_server_pids")


def _save_pids() -> None:
    with open(_pid_file(), "w") as f:
        f.write("\n".join(map(str, _server_pids)))


def _load_pids() -> list[int]:
    try:
        with open(_pid_file()) as f:
            return [int(x.strip()) for x in f if x.strip().isdigit()]
    except FileNotFoundError:
        return []


def main() -> None:
    global _server_pids
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1].lower()

    if cmd == "start":
        start_servers()
        _save_pids()
        print("Registering FastAPI agents with orchestrator...")
        register_all_agents_with_master()
        print("Servers are running. Use 'queries' to run load test, 'stop' to kill.")
        return

    if cmd == "stop":
        _server_pids = _load_pids()
        stop_servers()
        try:
            os.remove(_pid_file())
        except FileNotFoundError:
            pass
        kill_all_load_servers_by_port()
        return

    if cmd == "queries":
        rest = sys.argv[2:]
        stress = "stress" in rest
        max_q = next((int(x) for x in rest if x.isdigit()), None)
        test_name = next((x for x in rest if not x.isdigit() and x != "stress"), None)
        qs = get_stress_queries(max_q or 50) if stress else LOAD_TEST_QUERIES[: (max_q or len(LOAD_TEST_QUERIES))]
        print(f"Running {len(qs)} queries (assuming servers + master are up)...")
        stats = asyncio.run(run_queries(concurrency=5, queries=qs))
        metrics = compute_metrics(qs, stats["results"], stats["elapsed_sec"])
        print(format_metrics_report(metrics))
        if test_name:
            path = save_results(test_name, qs, stats, metrics)
            print(f"Results saved to {path}")
        for i, (status, val) in enumerate(stats["results"]):
            snippet = str(val)[:60] + "..." if len(str(val)) > 60 else val
            print(f"  [{i+1}] {status}: {snippet}")
        return

    if cmd == "run":
        test_name = sys.argv[2] if len(sys.argv) > 2 else None
        if not test_name:
            print("Usage: run <test_name>  (e.g. run baseline_001)", file=sys.stderr)
            sys.exit(1)
        start_servers()
        _save_pids()
        print("Registering FastAPI agents with orchestrator...")
        register_all_agents_with_master()
        try:
            print("Running load-test queries...")
            stats = asyncio.run(run_queries(concurrency=5))
            metrics = compute_metrics(LOAD_TEST_QUERIES, stats["results"], stats["elapsed_sec"])
            print(format_metrics_report(metrics))
            path = save_results(test_name, LOAD_TEST_QUERIES, stats, metrics)
            print(f"Results saved to {path}")
            for i, (status, val) in enumerate(stats["results"][:10]):
                snippet = str(val)[:50] + "..." if len(str(val)) > 50 else val
                print(f"  [{i+1}] {status}: {snippet}")
            if len(stats["results"]) > 10:
                print("  ...")
        finally:
            _server_pids = _load_pids()
            stop_servers()
            try:
                os.remove(_pid_file())
            except FileNotFoundError:
                pass
            kill_all_load_servers_by_port()
        return

    print("Unknown command. Use start | stop | queries | run")
    sys.exit(1)


if __name__ == "__main__":
    main()
