"""
Complex queries for orchestration and load testing.
Tuned for 10 complex apps: math, weather, text, time, data, code, finance, stats, search, workflow.
Includes metrics to analyze and evaluate results.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

# Queries that map to complex app tools (orchestrator picks the right agent)
SINGLE_TOOL_QUERIES = [
    "Add the numbers 17 and 25.",
    "What's the current weather in London?",
    "Reverse the string 'orchestration'.",
    "Convert 'hello world' to uppercase.",
    "What is the current time in UTC?",
    "Multiply 6 by 7.",
    "Get the GCD of 48 and 18.",
    "What is 12 factorial?",
    "Get a 3-day forecast for Paris.",
    "Slugify the text 'Hello World Example'.",
    "Extract all numbers from the string 'foo 12 bar 3.5 baz'.",
    "Sort the list [3, 1, 4, 1, 5] in ascending order.",
    "Compute SHA256 hash of 'hello'.",
    "Generate a new UUID.",
    "What is compound interest on $1000 at 5% for 10 years?",
    "What is the mean of [10, 20, 30, 40, 50]?",
    "Search for 'machine learning' and return 3 results.",
]

# Slightly ambiguous or multi-step phrasing (orchestrator picks one tool)
COMPLEX_QUERIES = [
    "I need the current time in UTC for a log.",
    "Compute the sum of 17 and 25.",
    "I'm planning a trip to Paris — what's the weather there?",
    "Take the string 'orchestration' and reverse it.",
    "Turn 'load test' into all caps.",
    "What is 6 times 7?",
    "What is the least common multiple of 4 and 6?",
    "Evaluate the polynomial with coefficients [1, 0, 2] at x=3 (i.e. 1 + 0*x + 2*x^2).",
    "Get humidity for Tokyo.",
    "Convert 25 Celsius to Fahrenheit.",
    "Filter the list [1, 5, 10, 15, 20] to keep values greater than 8.",
    "Base64 encode the string 'hello'.",
    "Parse the JSON string '{\"a\": 1}'.",
    "What is the monthly payment for a $200000 loan at 4% for 30 years?",
    "What is the median of [1, 3, 3, 6, 7, 8, 9]?",
    "Run a search query for 'orchestration' with limit 5.",
]

# Stress: many similar queries to flood the orchestrator
STRESS_QUERIES = [
    "What time is it?",
    "Echo: one.",
    "Add 1 and 1.",
] * 10  # 30 queries

# All combined for full load test
def get_all_queries(max_per_category: int | None = None) -> list[str]:
    out = []
    for q in SINGLE_TOOL_QUERIES + COMPLEX_QUERIES:
        if max_per_category and len(out) >= max_per_category:
            break
        out.append(q)
    return out


def get_stress_queries(n: int = 50) -> list[str]:
    return STRESS_QUERIES[:n]


# --- Metrics and evaluation ---

def _classify_error(msg: str) -> str:
    """Bucket error messages for breakdown."""
    if not msg:
        return "unknown"
    m = str(msg).lower()
    if "timeout" in m or "timed out" in m:
        return "timeout"
    if "connection" in m or "refused" in m or "connect" in m:
        return "connection"
    if "parse" in m or "json" in m:
        return "parse"
    if "unknown" in m or "not found" in m:
        return "unknown_agent_or_tool"
    if "invocation" in m or "failed" in m:
        return "invocation_failed"
    return "other"


def _result_looks_valid(status: str, value: Any) -> bool:
    """Heuristic: ok results that look like a real tool response (dict with content)."""
    if status != "ok":
        return False
    if value is None:
        return False
    if isinstance(value, dict):
        return len(value) > 0
    if isinstance(value, str):
        return len(value.strip()) > 0
    return True


# Optional: for a subset of queries, expected key or value to check correctness (query index -> check)
# Check can be a key that must exist, or (key, expected_value), or callable(result) -> bool.
EXPECTED_RESULT_CHECKS: dict[int, str | tuple | Any] = {
    0: "result",   # Add 17 and 25 -> result or sum
    1: "city",     # weather
    4: "iso",      # time UTC
    5: "result",   # multiply
    7: "result",   # factorial
    14: "amount",   # compound interest returns "amount" (not "result")
    15: "mean",    # mean of list
}


def _check_correctness(idx: int, result: Any) -> bool | None:
    """Return True if result matches expected for query idx, False if wrong, None if no check defined."""
    check = EXPECTED_RESULT_CHECKS.get(idx)
    if check is None:
        return None
    if not isinstance(result, dict):
        return False
    if callable(check):
        return bool(check(result))
    if isinstance(check, tuple):
        key, expected = check
        return result.get(key) == expected
    return check in result and result[check] is not None


def compute_metrics(
    queries: list[str],
    results: list[tuple[str, Any]],
    elapsed_sec: float,
) -> dict[str, Any]:
    """
    Analyze results and return metrics dict.
    results: list of (status, value) e.g. ("ok", {...}) or ("error", "message").
    """
    total = len(results)
    ok = sum(1 for s, _ in results if s == "ok")
    errors = total - ok
    error_values = [v for s, v in results if s != "ok"]
    error_classes = Counter(_classify_error(str(v)) for v in error_values)

    valid_results = sum(1 for s, v in results if _result_looks_valid(s, v))
    correctness_checks = 0
    correctness_pass = 0
    for i, (status, value) in enumerate(results):
        if i >= len(queries):
            break
        c = _check_correctness(i, value if status == "ok" else None)
        if c is not None:
            correctness_checks += 1
            if c:
                correctness_pass += 1

    return {
        "total": total,
        "ok": ok,
        "errors": errors,
        "success_rate": round(ok / total, 4) if total else 0,
        "error_rate": round(errors / total, 4) if total else 0,
        "elapsed_sec": round(elapsed_sec, 2),
        "qps": round(total / elapsed_sec, 2) if elapsed_sec > 0 else 0,
        "valid_result_rate": round(valid_results / total, 4) if total else 0,
        "error_breakdown": dict(error_classes),
        "correctness_checks": correctness_checks,
        "correctness_pass": correctness_pass,
        "correctness_rate": round(correctness_pass / correctness_checks, 4) if correctness_checks else None,
    }


def format_metrics_report(metrics: dict[str, Any], verbose: bool = True) -> str:
    """Human-readable metrics report."""
    lines = [
        "--- Metrics ---",
        f"  Total: {metrics['total']}  OK: {metrics['ok']}  Errors: {metrics['errors']}",
        f"  Success rate: {metrics['success_rate']:.2%}  Error rate: {metrics['error_rate']:.2%}",
        f"  Elapsed: {metrics['elapsed_sec']}s  QPS: {metrics['qps']}",
        f"  Valid result rate (non-empty): {metrics.get('valid_result_rate', 0):.2%}",
    ]
    if metrics.get("correctness_checks"):
        lines.append(
            f"  Correctness (sampled): {metrics['correctness_pass']}/{metrics['correctness_checks']} "
            f"({metrics.get('correctness_rate') or 0:.2%})"
        )
    if metrics.get("error_breakdown"):
        lines.append("  Error breakdown:")
        for k, v in sorted(metrics["error_breakdown"].items(), key=lambda x: -x[1]):
            lines.append(f"    {k}: {v}")
    return "\n".join(lines)
