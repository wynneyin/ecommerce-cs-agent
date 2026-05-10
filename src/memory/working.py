"""Working memory helpers — currently the per-turn dict in the AgentState.

This module is intentionally tiny; its purpose is to centralise the *concept*
so callers do not need to know which AgentState key holds working memory.
"""

from __future__ import annotations

from typing import Any

from src.state import AgentState


def get_working(state: AgentState) -> dict[str, Any]:
    return dict(state.get("memory_working") or {})


def update_working(state: AgentState, **kv: Any) -> dict[str, Any]:
    cur = get_working(state)
    cur.update(kv)
    return cur
