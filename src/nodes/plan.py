"""Plan node — produces the next-step tool plan.

Behaviour by mode:
* ``deterministic`` — direct mapping (intent + slots) -> single ToolCall.
* ``react`` — same baseline plan; if reflection requested replan, the slot
  inputs may have been edited by the reflect node.
"""

from __future__ import annotations

from src.llm import plan_for_intent
from src.state import AgentState
from src.trace import traced_node


@traced_node("plan")
def plan_node(state: AgentState) -> dict:
    intent = state.get("intent") or "unknown"
    slots = dict(state.get("slots") or {})
    # Inject query for searches / FAQ / web
    if intent in {"product_search", "faq_policy", "web_search"}:
        slots.setdefault("query", state.get("user_input") or "")
    # For memory_recall, plan stays empty; final response comes from memory
    plan_obj = plan_for_intent(intent, slots)
    plan = plan_obj.get("plan", [])
    return {"plan": plan}
