"""Reflect node — checks whether the actions satisfied the request.

If not, sets ``need_replan`` so ReAct loops back to plan. We cap replans to
avoid infinite loops.
"""

from __future__ import annotations

from src.state import AgentState
from src.trace import traced_node


MAX_REPLAN = 1


@traced_node("reflect")
def reflect_node(state: AgentState) -> dict:
    actions = state.get("actions") or []
    intent = state.get("intent")
    slots = state.get("slots") or {}
    replan_count = int(state.get("replan_count") or 0)

    last_ok = bool(actions and actions[-1].get("ok"))

    need_replan = False
    reason = "ok" if last_ok else "last_action_failed"

    # Intent-specific health check
    if intent == "product_compare":
        last = actions[-1].get("output") if actions else None
        if not last or not (isinstance(last, dict) and last.get("items") and len(last["items"]) >= 2):
            if replan_count < MAX_REPLAN and len(slots.get("product_ids") or []) >= 2:
                need_replan = True
                reason = "compare_insufficient_items"

    if intent == "order_query":
        last = actions[-1].get("output") if actions else None
        if not last or not (isinstance(last, dict) and last.get("ok")):
            if replan_count < MAX_REPLAN:
                # No useful replan target without more info → just record reflection
                reason = "order_not_found"

    if need_replan:
        replan_count += 1

    return {
        "reflection": reason,
        "need_replan": need_replan,
        "replan_count": replan_count,
    }
