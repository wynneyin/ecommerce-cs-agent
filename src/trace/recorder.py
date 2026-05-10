"""Run-level trace recorder.

Implements:
* `traced_node(name)` decorator that wraps a graph node and appends a
  structured `TraceEvent` to `state["trace"]` for both `start` and `end`.
* `format_trace_tree(events)` for pretty UI/Markdown rendering.
"""

from __future__ import annotations

import functools
import time
from datetime import datetime
from typing import Any, Callable

from src.state import AgentState, TraceEvent


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def _summarise(payload: Any) -> Any:
    """Reduce large structures so the trace stays readable."""
    if isinstance(payload, dict):
        out = {}
        for k, v in payload.items():
            if k in {"messages", "trace"}:
                continue
            if isinstance(v, list) and len(v) > 5:
                out[k] = {"_truncated_list": True, "len": len(v), "head": v[:3]}
            else:
                out[k] = v
        return out
    return payload


def traced_node(name: str) -> Callable:
    """Decorator: record start/end events around a LangGraph node function."""

    def decorator(func: Callable[[AgentState], dict]) -> Callable[[AgentState], dict]:
        @functools.wraps(func)
        def wrapper(state: AgentState) -> dict:
            t0 = time.perf_counter()
            start_event: TraceEvent = {
                "node": name,
                "phase": "start",
                "timestamp": _now(),
                "payload": _summarise(
                    {
                        "intent": state.get("intent"),
                        "slots": state.get("slots"),
                        "user_input": state.get("user_input"),
                    }
                ),
                "latency_ms": 0.0,
            }
            try:
                update = func(state) or {}
            except Exception as exc:  # pragma: no cover - safety net
                latency = (time.perf_counter() - t0) * 1000
                err_event: TraceEvent = {
                    "node": name,
                    "phase": "error",
                    "timestamp": _now(),
                    "payload": {"error": repr(exc)},
                    "latency_ms": latency,
                }
                existing = list(state.get("trace") or [])
                return {"trace": existing + [start_event, err_event]}

            latency = (time.perf_counter() - t0) * 1000
            end_event: TraceEvent = {
                "node": name,
                "phase": "end",
                "timestamp": _now(),
                "payload": _summarise(update),
                "latency_ms": round(latency, 2),
            }

            existing = list(state.get("trace") or [])
            new_trace = existing + [start_event, end_event]
            update = dict(update)
            update["trace"] = new_trace
            return update

        return wrapper

    return decorator


def format_trace_tree(events: list[TraceEvent]) -> str:
    """Render a compact tree-like view for CLI / Markdown output."""
    lines: list[str] = []
    for ev in events:
        marker = {"start": "▶", "end": "■", "error": "✖"}.get(ev.get("phase", "end"), "·")
        lat = ev.get("latency_ms") or 0.0
        node = ev.get("node", "?")
        suffix = f" ({lat:.1f} ms)" if ev.get("phase") == "end" else ""
        lines.append(f"{marker} {node}{suffix}")
    return "\n".join(lines)
