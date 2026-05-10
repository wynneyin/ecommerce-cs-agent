"""Router functions used by both deterministic and ReAct graphs."""

from __future__ import annotations

from src.config import SETTINGS
from src.state import AgentState


# After guardrails — block / continue
def after_guardrails(state: AgentState) -> str:
    return "continue" if state.get("guardrails_pass") else "block"


# After nlu — choose graph branch
def after_nlu(state: AgentState) -> str:
    intent = state.get("intent") or "unknown"
    if intent in {"unsafe", "smalltalk"}:
        return "shortcut"
    # 允许「未分类」走 think→plan，由模型决定是否仅调用 web_search
    if intent == "unknown" and not SETTINGS.use_llm_web_tool():
        return "shortcut"
    if intent == "unknown" and SETTINGS.use_llm_web_tool():
        return "plan"
    if intent == "memory_recall":
        return "shortcut"
    if intent == "faq_policy":
        return "retrieve"  # retrieval-first
    if intent in {"product_search", "product_detail", "product_compare"}:
        return "retrieve"
    if intent in {"order_query", "refund_request", "web_search"}:
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
