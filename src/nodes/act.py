"""Act node — executes the planned tool calls."""

from __future__ import annotations

import time
from typing import Any

from src.state import AgentState, ActionRecord
from src.tools import SENSITIVE_TOOLS, get_tool
from src.trace import traced_node


def _exec(name: str, args: dict[str, Any]) -> tuple[Any, bool, str | None, float]:
    fn = get_tool(name)
    if fn is None:
        return None, False, f"unknown tool: {name}", 0.0
    t0 = time.perf_counter()
    try:
        out = fn(**(args or {}))
    except TypeError as exc:
        return None, False, f"bad args: {exc!r}", (time.perf_counter() - t0) * 1000
    except Exception as exc:  # pragma: no cover
        return None, False, repr(exc), (time.perf_counter() - t0) * 1000
    latency = (time.perf_counter() - t0) * 1000
    ok = bool(out and (out.get("ok", True) if isinstance(out, dict) else True))
    err = None if ok else (out.get("error") if isinstance(out, dict) else "tool error")
    return out, ok, err, latency


@traced_node("act")
def act_node(state: AgentState) -> dict:
    plan = list(state.get("plan") or [])
    actions: list[ActionRecord] = list(state.get("actions") or [])

    confirm_required = bool(state.get("confirm_required"))
    confirm_payload = state.get("confirm_payload")
    confirm_decision = state.get("confirm_decision")

    for tool_call in plan:
        name = tool_call.get("name")
        args = dict(tool_call.get("args") or {})

        if not name:
            continue

        # If the tool is sensitive and not yet confirmed → emit a confirm prompt
        if name in SENSITIVE_TOOLS and confirm_decision != "approve":
            preview, _, _, latency = _exec(name, args)  # dry-run produces preview payload
            if isinstance(preview, dict) and preview.get("needs_confirmation"):
                # Record the dry-run so traces / metrics see the tool call (with
                # ok=False because it's pending confirmation).
                actions.append(
                    ActionRecord(
                        name=name,
                        args=args,
                        output=preview,
                        ok=False,
                        error="needs_confirmation",
                        latency_ms=round(latency, 2),
                    )
                )
                return {
                    "confirm_required": True,
                    "confirm_payload": {
                        "tool": name,
                        "args": args,
                        "preview": preview.get("preview"),
                    },
                    "actions": actions,
                }

        # Sensitive + confirmed → re-call with confirmed flag
        if name in SENSITIVE_TOOLS and confirm_decision == "approve":
            args = {**args, "confirmed": True}
            confirm_required = False

        # Sensitive + rejected → record failure and short-circuit
        if name in SENSITIVE_TOOLS and confirm_decision == "reject":
            actions.append(
                ActionRecord(
                    name=name,
                    args=args,
                    output={"ok": False, "error": "user rejected"},
                    ok=False,
                    error="user rejected",
                    latency_ms=0.0,
                )
            )
            confirm_required = False
            break

        out, ok, err, latency = _exec(name, args)
        actions.append(
            ActionRecord(
                name=name,
                args=args,
                output=out,
                ok=ok,
                error=err,
                latency_ms=round(latency, 2),
            )
        )

    return {
        "actions": actions,
        "confirm_required": confirm_required,
        "confirm_payload": confirm_payload if confirm_required else None,
    }
