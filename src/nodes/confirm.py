"""Confirm node — placeholder where the graph interrupts before sensitive ops.

In practice the graph uses ``interrupt_before=["act"]`` (or a dedicated branch)
when ``confirm_required`` is True. This node only formats the human-facing
prompt and synchronises the state when an external decision is supplied.
"""

from __future__ import annotations

from src.state import AgentState
from src.trace import traced_node


@traced_node("confirm")
def confirm_node(state: AgentState) -> dict:
    payload = state.get("confirm_payload") or {}
    preview = payload.get("preview") if isinstance(payload, dict) else None
    msg = (
        preview.get("warning") if isinstance(preview, dict) else None
    ) or "该操作需要您确认才能继续。"
    return {
        "final_response": f"[需要确认] {msg}",
        "confirm_required": True,
    }
