"""Router functions used by both deterministic and ReAct graphs."""

from __future__ import annotations

from src.state import AgentState


# After guardrails — block / continue
def after_guardrails(state: AgentState) -> str:
    return "continue" if state.get("guardrails_pass") else "block"


# After nlu — choose graph branch
def after_nlu(state: AgentState) -> str:
    intent = state.get("intent") or "unknown"
    if intent in {"unsafe", "smalltalk", "unknown"}:
        return "shortcut"
    if intent == "memory_recall":
        return "shortcut"
    if intent == "faq_policy":
        return "retrieve"  # retrieval-first
    if intent in {"product_search", "product_detail", "product_compare"}:
        return "retrieve"
    if intent in {"order_query", "refund_request"}:
        return "plan"  # tool-first, no retrieval
    return "shortcut"


# After act — confirm? reflect?
def after_act(state: AgentState) -> str:
    if state.get("confirm_required"):
        return "confirm"
    return "reflect"


# After reflect — replan / finish
def after_reflect(state: AgentState) -> str:
    if state.get("need_replan"):
        return "replan"
    return "finish"
