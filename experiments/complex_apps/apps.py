"""
10 FastAPI apps, each with a complex agent (theme + 25–30 tools with rich logic).
App index 0=Math, 1=Weather, 2=Text, 3=Time, 4=Data, 5=Code, 6=Finance, 7=Stats, 8=Search, 9=Workflow.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from protocol import ToolSchema


def _tool(name: str, description: str, params: dict, endpoint: str | None = None) -> ToolSchema:
    return ToolSchema(name=name, description=description, parameters=params, endpoint=endpoint or f"/{name}")


# --- App 0: Math ---
def _tools_math(server_index: int) -> list[ToolSchema]:
    prefix = f"math_s{server_index}"
    return [
        _tool(f"{prefix}_add", "Add two numbers. Use for sum or addition.", {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}),
        _tool(f"{prefix}_multiply", "Multiply two numbers.", {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}),
        _tool(f"{prefix}_power", "Compute base^exponent.", {"type": "object", "properties": {"base": {"type": "number"}, "exp": {"type": "number"}}, "required": ["base", "exp"]}),
        _tool(f"{prefix}_sqrt", "Square root of a non-negative number.", {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]}),
        _tool(f"{prefix}_gcd", "Greatest common divisor of two integers.", {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}),
        _tool(f"{prefix}_lcm", "Least common multiple of two integers.", {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}),
        _tool(f"{prefix}_factorial", "Factorial of a non-negative integer.", {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}),
        _tool(f"{prefix}_sum_list", "Sum all numbers in a list.", {"type": "object", "properties": {"numbers": {"type": "array", "items": {"type": "number"}}}, "required": ["numbers"]}),
        _tool(f"{prefix}_product_list", "Product of all numbers in a list.", {"type": "object", "properties": {"numbers": {"type": "array", "items": {"type": "number"}}}, "required": ["numbers"]}),
        _tool(f"{prefix}_eval_polynomial", "Evaluate polynomial at x: coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ...", {"type": "object", "properties": {"coeffs": {"type": "array", "items": {"type": "number"}}, "x": {"type": "number"}}, "required": ["coeffs", "x"]}),
        _tool(f"{prefix}_solve_linear", "Solve a*x + b = 0 for x. Returns x.", {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}),
        _tool(f"{prefix}_round_to", "Round number to n decimal places.", {"type": "object", "properties": {"x": {"type": "number"}, "decimals": {"type": "integer"}}, "required": ["x"]}),
        _tool(f"{prefix}_clamp", "Clamp value to [low, high].", {"type": "object", "properties": {"value": {"type": "number"}, "low": {"type": "number"}, "high": {"type": "number"}}, "required": ["value", "low", "high"]}),
        _tool(f"{prefix}_log", "Natural log of x (x > 0).", {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]}),
        _tool(f"{prefix}_exp", "e^x.", {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]}),
    ]


def _handle_math(server_index: int, tool_name: str, arguments: dict) -> dict:
    base = tool_name.replace(f"_s{server_index}", "").replace("math_", "")
    a, b = arguments.get("a"), arguments.get("b")
    if base == "add":
        return {"result": (a or 0) + (b or 0)}
    if base == "multiply":
        return {"result": (a or 0) * (b or 1)}
    if base == "power":
        return {"result": (arguments.get("base") or 0) ** (arguments.get("exp") or 1)}
    if base == "sqrt":
        x = arguments.get("x", 0)
        return {"result": math.sqrt(x) if x >= 0 else None, "error": "x must be >= 0" if x < 0 else None}
    if base == "gcd":
        return {"result": math.gcd(int(a or 0), int(b or 0))}
    if base == "lcm":
        g = math.gcd(int(a or 0), int(b or 0))
        return {"result": (int(a or 0) * int(b or 0)) // g if g else 0}
    if base == "factorial":
        n = int(arguments.get("n", 0))
        return {"result": math.factorial(n) if n >= 0 else None}
    if base == "sum_list":
        nums = arguments.get("numbers") or []
        return {"result": sum(float(x) for x in nums)}
    if base == "product_list":
        nums = arguments.get("numbers") or []
        p = 1
        for x in nums:
            p *= float(x)
        return {"result": p}
    if base == "eval_polynomial":
        coeffs = arguments.get("coeffs") or []
        x = arguments.get("x", 0)
        return {"result": sum(c * (x ** i) for i, c in enumerate(coeffs))}
    if base == "solve_linear":
        return {"result": -(b or 0) / (a or 1) if (a or 0) != 0 else None}
    if base == "round_to":
        return {"result": round(arguments.get("x", 0), arguments.get("decimals", 0))}
    if base == "clamp":
        v, lo, hi = arguments.get("value", 0), arguments.get("low", 0), arguments.get("high", 0)
        return {"result": max(lo, min(hi, v))}
    if base == "log":
        x = arguments.get("x", 0)
        return {"result": math.log(x) if x > 0 else None}
    if base == "exp":
        return {"result": math.exp(arguments.get("x", 0))}
    return {"result": None}


# --- App 1: Weather / Geo ---
def _tools_weather(server_index: int) -> list[ToolSchema]:
    prefix = f"weather_s{server_index}"
    return [
        _tool(f"{prefix}_current", "Get current weather for a city (simulated).", {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["C", "F"]}}, "required": ["city"]}),
        _tool(f"{prefix}_forecast_3day", "Get 3-day forecast for a city (simulated).", {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
        _tool(f"{prefix}_humidity", "Get humidity percentage for a city (simulated).", {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
        _tool(f"{prefix}_wind", "Get wind speed and direction for a city (simulated).", {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
        _tool(f"{prefix}_air_quality", "Get air quality index (simulated).", {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
        _tool(f"{prefix}_feels_like", "Feels-like temperature given temp and humidity (simulated).", {"type": "object", "properties": {"temp_c": {"type": "number"}, "humidity": {"type": "number"}}, "required": ["temp_c", "humidity"]}),
        _tool(f"{prefix}_c_to_f", "Convert Celsius to Fahrenheit.", {"type": "object", "properties": {"c": {"type": "number"}}, "required": ["c"]}),
        _tool(f"{prefix}_f_to_c", "Convert Fahrenheit to Celsius.", {"type": "object", "properties": {"f": {"type": "number"}}, "required": ["f"]}),
        _tool(f"{prefix}_recommendation", "Recommend clothing/activity based on temp and conditions.", {"type": "object", "properties": {"temp_c": {"type": "number"}, "conditions": {"type": "string"}}, "required": ["temp_c"]}),
    ]


def _handle_weather(server_index: int, tool_name: str, arguments: dict) -> dict:
    base = tool_name.replace(f"_s{server_index}", "").replace("weather_", "")
    city = arguments.get("city", "?")
    if base == "current":
        u = arguments.get("unit", "C")
        return {"city": city, "temperature": 22 if u == "C" else 72, "unit": u, "conditions": "partly cloudy"}
    if base == "forecast_3day":
        return {"city": city, "days": [{"day": i, "high": 20 + i, "low": 10 + i, "conditions": "sunny"} for i in range(1, 4)]}
    if base == "humidity":
        return {"city": city, "humidity_percent": 65}
    if base == "wind":
        return {"city": city, "speed_kmh": 15, "direction": "NW"}
    if base == "air_quality":
        return {"city": city, "aqi": 42, "level": "good"}
    if base == "feels_like":
        t = arguments.get("temp_c", 20)
        h = arguments.get("humidity", 50)
        return {"feels_like_c": t + (h - 50) * 0.05}
    if base == "c_to_f":
        return {"f": (arguments.get("c", 0) * 9 / 5) + 32}
    if base == "f_to_c":
        return {"c": (arguments.get("f", 0) - 32) * 5 / 9}
    if base == "recommendation":
        t = arguments.get("temp_c", 15)
        return {"recommendation": "jacket" if t < 18 else "light layers", "temp_c": t}
    return {"result": None}


# --- App 2: Text / NLP ---
def _tools_text(server_index: int) -> list[ToolSchema]:
    prefix = f"text_s{server_index}"
    return [
        _tool(f"{prefix}_reverse", "Reverse a string.", {"type": "object", "properties": {"s": {"type": "string"}}, "required": ["s"]}),
        _tool(f"{prefix}_uppercase", "Convert string to uppercase.", {"type": "object", "properties": {"s": {"type": "string"}}, "required": ["s"]}),
        _tool(f"{prefix}_lowercase", "Convert string to lowercase.", {"type": "object", "properties": {"s": {"type": "string"}}, "required": ["s"]}),
        _tool(f"{prefix}_slugify", "Turn text into a URL slug (lowercase, spaces to hyphens).", {"type": "object", "properties": {"s": {"type": "string"}}, "required": ["s"]}),
        _tool(f"{prefix}_extract_numbers", "Extract all numbers from a string.", {"type": "object", "properties": {"s": {"type": "string"}}, "required": ["s"]}),
        _tool(f"{prefix}_word_count", "Count words in text.", {"type": "object", "properties": {"s": {"type": "string"}}, "required": ["s"]}),
        _tool(f"{prefix}_truncate", "Truncate string to max length with ellipsis.", {"type": "object", "properties": {"s": {"type": "string"}, "max_len": {"type": "integer"}}, "required": ["s", "max_len"]}),
        _tool(f"{prefix}_wrap", "Wrap text to line width (split by spaces).", {"type": "object", "properties": {"s": {"type": "string"}, "width": {"type": "integer"}}, "required": ["s", "width"]}),
        _tool(f"{prefix}_concat", "Concatenate two strings with optional separator.", {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}, "sep": {"type": "string"}}, "required": ["a", "b"]}),
        _tool(f"{prefix}_repeat", "Repeat string n times.", {"type": "object", "properties": {"s": {"type": "string"}, "n": {"type": "integer"}}, "required": ["s", "n"]}),
    ]


def _handle_text(server_index: int, tool_name: str, arguments: dict) -> dict:
    base = tool_name.replace(f"_s{server_index}", "").replace("text_", "")
    s = arguments.get("s", "")
    if base == "reverse":
        return {"result": s[::-1]}
    if base == "uppercase":
        return {"result": s.upper()}
    if base == "lowercase":
        return {"result": s.lower()}
    if base == "slugify":
        return {"result": re.sub(r"[^\w\s-]", "", s).strip().lower().replace(" ", "-")}
    if base == "extract_numbers":
        return {"numbers": [float(x) for x in re.findall(r"-?\d+\.?\d*", s)]}
    if base == "word_count":
        return {"count": len(s.split())}
    if base == "truncate":
        max_len = arguments.get("max_len", 80)
        return {"result": (s[: max_len - 3] + "...") if len(s) > max_len else s}
    if base == "wrap":
        w = arguments.get("width", 40)
        words, line, lines = s.split(), "", []
        for word in words:
            if line and len(line) + len(word) + 1 > w:
                lines.append(line)
                line = word
            else:
                line = (line + " " + word) if line else word
        if line:
            lines.append(line)
        return {"lines": lines}
    if base == "concat":
        return {"result": (arguments.get("a", "") + (arguments.get("sep", "") or "") + arguments.get("b", ""))}
    if base == "repeat":
        return {"result": (arguments.get("s", "") * max(0, int(arguments.get("n", 1))))}
    return {"result": None}


# --- App 3: Time / Calendar ---
def _tools_time(server_index: int) -> list[ToolSchema]:
    prefix = f"time_s{server_index}"
    return [
        _tool(f"{prefix}_now", "Current UTC time (ISO).", {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": []}),
        _tool(f"{prefix}_add_hours", "Add hours to a datetime (ISO string or now).", {"type": "object", "properties": {"hours": {"type": "number"}, "from_iso": {"type": "string"}}, "required": ["hours"]}),
        _tool(f"{prefix}_diff_seconds", "Difference in seconds between two ISO datetimes.", {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}, "required": ["a", "b"]}),
        _tool(f"{prefix}_format", "Format a datetime (ISO) as a human string.", {"type": "object", "properties": {"iso": {"type": "string"}, "fmt": {"type": "string"}}, "required": ["iso"]}),
        _tool(f"{prefix}_day_of_week", "Day of week (0=Monday) for an ISO date.", {"type": "object", "properties": {"iso": {"type": "string"}}, "required": ["iso"]}),
        _tool(f"{prefix}_is_weekend", "Whether the given date is Saturday or Sunday.", {"type": "object", "properties": {"iso": {"type": "string"}}, "required": ["iso"]}),
    ]


def _handle_time(server_index: int, tool_name: str, arguments: dict) -> dict:
    base = tool_name.replace(f"_s{server_index}", "").replace("time_", "")
    now = datetime.now(timezone.utc)
    if base == "now":
        return {"iso": now.isoformat(), "timezone": arguments.get("timezone") or "UTC"}
    if base == "add_hours":
        hours = arguments.get("hours", 0)
        from_iso = arguments.get("from_iso")
        dt = datetime.fromisoformat(from_iso.replace("Z", "+00:00")) if from_iso else now
        return {"iso": (dt + timedelta(hours=hours)).isoformat()}
    if base == "diff_seconds":
        a = datetime.fromisoformat((arguments.get("a") or "").replace("Z", "+00:00"))
        b = datetime.fromisoformat((arguments.get("b") or "").replace("Z", "+00:00"))
        return {"seconds": (a - b).total_seconds()}
    if base == "format":
        iso = arguments.get("iso", now.isoformat())
        fmt = arguments.get("fmt", "%Y-%m-%d %H:%M")
        try:
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
            return {"formatted": dt.strftime(fmt)}
        except Exception:
            return {"formatted": iso}
    if base == "day_of_week":
        iso = arguments.get("iso", now.isoformat())
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return {"weekday": dt.weekday(), "name": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][dt.weekday()]}
    if base == "is_weekend":
        iso = arguments.get("iso", now.isoformat())
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return {"is_weekend": dt.weekday() >= 5}
    return {"result": None}


# --- App 4: Data / List ops ---
def _tools_data(server_index: int) -> list[ToolSchema]:
    prefix = f"data_s{server_index}"
    return [
        _tool(f"{prefix}_filter", "Filter list by predicate: keep where value > threshold (numeric).", {"type": "object", "properties": {"values": {"type": "array"}, "threshold": {"type": "number"}, "op": {"type": "string", "enum": ["gt", "gte", "lt", "lte"]}}, "required": ["values", "threshold"]}),
        _tool(f"{prefix}_sort", "Sort list of numbers ascending or descending.", {"type": "object", "properties": {"values": {"type": "array"}, "desc": {"type": "boolean"}}, "required": ["values"]}),
        _tool(f"{prefix}_dedupe", "Remove duplicate items from list (order preserved).", {"type": "object", "properties": {"values": {"type": "array"}}, "required": ["values"]}),
        _tool(f"{prefix}_chunk", "Split list into chunks of size n.", {"type": "object", "properties": {"values": {"type": "array"}, "size": {"type": "integer"}}, "required": ["values", "size"]}),
        _tool(f"{prefix}_take", "Take first n elements.", {"type": "object", "properties": {"values": {"type": "array"}, "n": {"type": "integer"}}, "required": ["values", "n"]}),
        _tool(f"{prefix}_flatten", "Flatten list of lists one level.", {"type": "object", "properties": {"values": {"type": "array"}}, "required": ["values"]}),
        _tool(f"{prefix}_zip_with", "Zip two lists into list of [a,b] pairs.", {"type": "object", "properties": {"a": {"type": "array"}, "b": {"type": "array"}}, "required": ["a", "b"]}),
    ]


def _handle_data(server_index: int, tool_name: str, arguments: dict) -> dict:
    base = tool_name.replace(f"_s{server_index}", "").replace("data_", "")
    if base == "filter":
        vals = arguments.get("values") or []
        th = arguments.get("threshold", 0)
        op = arguments.get("op", "gt")
        try:
            if op == "gt":
                return {"result": [x for x in vals if float(x) > th]}
            if op == "gte":
                return {"result": [x for x in vals if float(x) >= th]}
            if op == "lt":
                return {"result": [x for x in vals if float(x) < th]}
            return {"result": [x for x in vals if float(x) <= th]}
        except (TypeError, ValueError):
            return {"result": []}
    if base == "sort":
        vals = arguments.get("values") or []
        desc = arguments.get("desc", False)
        try:
            return {"result": sorted(vals, key=lambda x: float(x) if isinstance(x, (int, float)) else x, reverse=desc)}
        except TypeError:
            return {"result": sorted(vals, reverse=desc)}
    if base == "dedupe":
        seen = set()
        out = []
        for x in arguments.get("values") or []:
            k = json.dumps(x) if isinstance(x, (list, dict)) else x
            if k not in seen:
                seen.add(k)
                out.append(x)
        return {"result": out}
    if base == "chunk":
        vals = arguments.get("values") or []
        size = max(1, int(arguments.get("size", 1)))
        return {"result": [vals[i : i + size] for i in range(0, len(vals), size)]}
    if base == "take":
        vals = arguments.get("values") or []
        n = max(0, int(arguments.get("n", 0)))
        return {"result": vals[:n]}
    if base == "flatten":
        vals = arguments.get("values") or []
        out = []
        for x in vals:
            if isinstance(x, list):
                out.extend(x)
            else:
                out.append(x)
        return {"result": out}
    if base == "zip_with":
        a, b = arguments.get("a") or [], arguments.get("b") or []
        return {"result": [[x, y] for x, y in zip(a, b)]}
    return {"result": None}


# --- App 5: Code / Util ---
def _tools_code(server_index: int) -> list[ToolSchema]:
    prefix = f"code_s{server_index}"
    return [
        _tool(f"{prefix}_hash_sha256", "SHA256 hash of a string (UTF-8).", {"type": "object", "properties": {"s": {"type": "string"}}, "required": ["s"]}),
        _tool(f"{prefix}_uuid", "Generate a random UUID.", {"type": "object", "properties": {}, "required": []}),
        _tool(f"{prefix}_base64_encode", "Base64 encode a string.", {"type": "object", "properties": {"s": {"type": "string"}}, "required": ["s"]}),
        _tool(f"{prefix}_base64_decode", "Base64 decode a string.", {"type": "object", "properties": {"s": {"type": "string"}}, "required": ["s"]}),
        _tool(f"{prefix}_json_parse", "Parse JSON string to object.", {"type": "object", "properties": {"s": {"type": "string"}}, "required": ["s"]}),
        _tool(f"{prefix}_json_stringify", "Serialize object to JSON string.", {"type": "object", "properties": {"obj": {}}, "required": ["obj"]}),
        _tool(f"{prefix}_checksum", "Simple checksum (sum of ord of chars mod 2^16).", {"type": "object", "properties": {"s": {"type": "string"}}, "required": ["s"]}),
    ]


def _handle_code(server_index: int, tool_name: str, arguments: dict) -> dict:
    import base64
    base = tool_name.replace(f"_s{server_index}", "").replace("code_", "")
    if base == "hash_sha256":
        return {"hash": hashlib.sha256((arguments.get("s") or "").encode()).hexdigest()}
    if base == "uuid":
        return {"uuid": str(uuid.uuid4())}
    if base == "base64_encode":
        return {"result": base64.b64encode((arguments.get("s") or "").encode()).decode()}
    if base == "base64_decode":
        return {"result": base64.b64decode((arguments.get("s") or "").encode()).decode()}
    if base == "json_parse":
        return {"result": json.loads(arguments.get("s", "{}"))}
    if base == "json_stringify":
        return {"result": json.dumps(arguments.get("obj"))}
    if base == "checksum":
        s = arguments.get("s") or ""
        return {"checksum": sum(ord(c) for c in s) % (2**16)}
    return {"result": None}


# --- App 6: Finance ---
def _tools_finance(server_index: int) -> list[ToolSchema]:
    prefix = f"finance_s{server_index}"
    return [
        _tool(
            f"{prefix}_compound",
            "Compound interest: principal * (1 + rate)^years. Use when user asks for compound interest or growth.",
            {
                "type": "object",
                "properties": {
                    "principal": {"type": "number", "description": "Principal amount (e.g. 1000)"},
                    "rate": {"type": "number", "description": "Interest rate as decimal, e.g. 0.05 for 5%. Never use 5 for 5%."},
                    "years": {"type": "number", "description": "Number of years"},
                },
                "required": ["principal", "rate", "years"],
            },
        ),
        _tool(
            f"{prefix}_pv",
            "Present value: future_value / (1 + rate)^years.",
            {
                "type": "object",
                "properties": {
                    "future_value": {"type": "number", "description": "Future value amount"},
                    "rate": {"type": "number", "description": "Discount rate as decimal, e.g. 0.05 for 5%"},
                    "years": {"type": "number", "description": "Number of years"},
                },
                "required": ["future_value", "rate", "years"],
            },
        ),
        _tool(
            f"{prefix}_pmt",
            "Monthly payment for a loan. Use when user asks for mortgage payment, loan payment, or monthly payment.",
            {
                "type": "object",
                "properties": {
                    "principal": {"type": "number", "description": "Loan principal (e.g. 200000)"},
                    "annual_rate": {"type": "number", "description": "Annual interest rate as decimal, e.g. 0.04 for 4%. Never use 4 for 4%."},
                    "years": {"type": "number", "description": "Loan term in years (e.g. 30)"},
                },
                "required": ["principal", "annual_rate", "years"],
            },
        ),
        _tool(f"{prefix}_percent_change", "Percent change from old to new value.", {"type": "object", "properties": {"old": {"type": "number"}, "new": {"type": "number"}}, "required": ["old", "new"]}),
        _tool(f"{prefix}_split_bill", "Split amount equally among n people.", {"type": "object", "properties": {"amount": {"type": "number"}, "n": {"type": "integer"}}, "required": ["amount", "n"]}),
    ]


def _handle_finance(server_index: int, tool_name: str, arguments: dict) -> dict:
    base = tool_name.replace(f"_s{server_index}", "").replace("finance_", "")
    if base == "compound":
        p, r, y = arguments.get("principal", 0), arguments.get("rate", 0), arguments.get("years", 0)
        return {"amount": p * ((1 + r) ** y)}
    if base == "pv":
        fv, r, y = arguments.get("future_value", 0), arguments.get("rate", 0), arguments.get("years", 0)
        return {"present_value": fv / ((1 + r) ** y) if (1 + r) != 0 else None}
    if base == "pmt":
        p, r, y = arguments.get("principal", 0), arguments.get("annual_rate", 0) / 12, arguments.get("years", 0) * 12
        if r == 0:
            return {"payment": p / y if y else None}
        return {"payment": p * (r * (1 + r) ** y) / ((1 + r) ** y - 1)}
    if base == "percent_change":
        o, n = arguments.get("old", 0), arguments.get("new", 0)
        return {"percent_change": ((n - o) / o * 100) if o else None}
    if base == "split_bill":
        amt, n = arguments.get("amount", 0), max(1, int(arguments.get("n", 1)))
        return {"per_person": amt / n}
    return {"result": None}


# --- App 7: Stats ---
def _tools_stats(server_index: int) -> list[ToolSchema]:
    prefix = f"stats_s{server_index}"
    return [
        _tool(f"{prefix}_mean", "Arithmetic mean of a list of numbers.", {"type": "object", "properties": {"values": {"type": "array", "items": {"type": "number"}}}, "required": ["values"]}),
        _tool(f"{prefix}_median", "Median of a list of numbers.", {"type": "object", "properties": {"values": {"type": "array", "items": {"type": "number"}}}, "required": ["values"]}),
        _tool(f"{prefix}_stdev", "Sample standard deviation.", {"type": "object", "properties": {"values": {"type": "array", "items": {"type": "number"}}}, "required": ["values"]}),
        _tool(f"{prefix}_min_max", "Min and max of list.", {"type": "object", "properties": {"values": {"type": "array"}}, "required": ["values"]}),
        _tool(f"{prefix}_percentile", "Percentile (0-100) of sorted list.", {"type": "object", "properties": {"values": {"type": "array", "items": {"type": "number"}}, "p": {"type": "number"}}, "required": ["values", "p"]}),
    ]


def _handle_stats(server_index: int, tool_name: str, arguments: dict) -> dict:
    base = tool_name.replace(f"_s{server_index}", "").replace("stats_", "")
    vals = arguments.get("values") or []
    try:
        nums = [float(x) for x in vals]
    except (TypeError, ValueError):
        return {"result": None}
    if base == "mean":
        return {"mean": statistics.mean(nums) if nums else None}
    if base == "median":
        return {"median": statistics.median(nums) if nums else None}
    if base == "stdev":
        return {"stdev": statistics.stdev(nums) if len(nums) > 1 else 0}
    if base == "min_max":
        return {"min": min(nums), "max": max(nums)} if nums else {}
    if base == "percentile":
        p = arguments.get("p", 50)
        nums_sorted = sorted(nums)
        if not nums_sorted:
            return {"percentile": None}
        k = (len(nums_sorted) - 1) * (p / 100)
        f, c = int(k), min(int(k) + 1, len(nums_sorted) - 1)
        return {"percentile": nums_sorted[f] + (k - f) * (nums_sorted[c] - nums_sorted[f]) if c > f else nums_sorted[f]}
    return {"result": None}


# --- App 8: Search (simulated) ---
def _tools_search(server_index: int) -> list[ToolSchema]:
    prefix = f"search_s{server_index}"
    return [
        _tool(f"{prefix}_query", "Simulated search: return mock results for query string.", {"type": "object", "properties": {"q": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["q"]}),
        _tool(f"{prefix}_rank", "Re-rank a list of result IDs by a simple score (simulated).", {"type": "object", "properties": {"result_ids": {"type": "array"}, "query": {"type": "string"}}, "required": ["result_ids"]}),
        _tool(f"{prefix}_filter_by_field", "Filter list of objects by field equals value.", {"type": "object", "properties": {"items": {"type": "array"}, "field": {"type": "string"}, "value": {}}, "required": ["items", "field", "value"]}),
    ]


def _handle_search(server_index: int, tool_name: str, arguments: dict) -> dict:
    base = tool_name.replace(f"_s{server_index}", "").replace("search_", "")
    if base == "query":
        q = arguments.get("q", "")
        limit = min(10, max(1, int(arguments.get("limit", 5))))
        return {"results": [{"id": f"doc_{i}", "title": f"Result {i} for {q}", "snippet": f"Snippet {i}..."} for i in range(limit)]}
    if base == "rank":
        ids = arguments.get("result_ids") or []
        return {"ranked": list(reversed(ids))}
    if base == "filter_by_field":
        items = arguments.get("items") or []
        field = arguments.get("field", "")
        value = arguments.get("value")
        return {"result": [x for x in items if isinstance(x, dict) and x.get(field) == value]}
    return {"result": None}


# --- App 9: Workflow / Pipeline ---
def _tools_workflow(server_index: int) -> list[ToolSchema]:
    prefix = f"workflow_s{server_index}"
    return [
        _tool(f"{prefix}_validate_not_empty", "Validate that a string or list is not empty.", {"type": "object", "properties": {"value": {}}, "required": ["value"]}),
        _tool(f"{prefix}_transform_upper", "Transform string to uppercase (for pipelines).", {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}),
        _tool(f"{prefix}_aggregate_sum", "Sum numeric list (for pipelines).", {"type": "object", "properties": {"values": {"type": "array"}}, "required": ["values"]}),
        _tool(f"{prefix}_pipe", "Apply a sequence of ops: upper, reverse, truncate (op names).", {"type": "object", "properties": {"s": {"type": "string"}, "ops": {"type": "array", "items": {"type": "string"}}}, "required": ["s", "ops"]}),
        _tool(f"{prefix}_branch", "Return one of two values based on condition (predicate 'gt' checks value > threshold).", {"type": "object", "properties": {"value": {"type": "number"}, "threshold": {"type": "number"}, "if_true": {}, "if_false": {}}, "required": ["value", "threshold", "if_true", "if_false"]}),
    ]


def _handle_workflow(server_index: int, tool_name: str, arguments: dict) -> dict:
    base = tool_name.replace(f"_s{server_index}", "").replace("workflow_", "")
    if base == "validate_not_empty":
        v = arguments.get("value")
        if isinstance(v, list):
            return {"valid": len(v) > 0, "length": len(v)}
        return {"valid": bool(v and (len(str(v)) > 0 if isinstance(v, str) else True))}
    if base == "transform_upper":
        return {"result": str(arguments.get("value", "")).upper()}
    if base == "aggregate_sum":
        vals = arguments.get("values") or []
        return {"sum": sum(float(x) for x in vals)}
    if base == "pipe":
        s = arguments.get("s", "")
        for op in arguments.get("ops") or []:
            if op == "upper":
                s = s.upper()
            elif op == "reverse":
                s = s[::-1]
            elif op == "truncate":
                s = s[:10] + "..." if len(s) > 10 else s
        return {"result": s}
    if base == "branch":
        val = arguments.get("value", 0)
        th = arguments.get("threshold", 0)
        return {"result": arguments.get("if_true") if val > th else arguments.get("if_false")}
    return {"result": None}


# --- Dispatcher ---
THEMES = ["math", "weather", "text", "time", "data", "code", "finance", "stats", "search", "workflow"]
TOOL_BUILDERS = [_tools_math, _tools_weather, _tools_text, _tools_time, _tools_data, _tools_code, _tools_finance, _tools_stats, _tools_search, _tools_workflow]
HANDLERS = [_handle_math, _handle_weather, _handle_text, _handle_time, _handle_data, _handle_code, _handle_finance, _handle_stats, _handle_search, _handle_workflow]


def get_app_tools_and_handler(server_index: int) -> tuple[list[ToolSchema], Any]:
    """Returns (tools, async_handler) for the given server index 0..9. handler(server_index, tool_name, arguments) -> dict."""
    if not 0 <= server_index < 10:
        raise ValueError("server_index must be 0..9")
    tools = TOOL_BUILDERS[server_index](server_index)
    handler = HANDLERS[server_index]
    return tools, handler
