"""NLU node — Chinese rule template first, LLM fallback for low confidence."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage

from src.config import SETTINGS
from src.llm import get_chat_model, run_nlu
from src.state import AgentState
from src.trace import traced_node


LLM_FALLBACK_THRESHOLD = 0.5


def _maybe_llm_refine(text: str, base: dict[str, Any]) -> dict[str, Any]:
    """Fall back to the chat model only if the rule-based confidence is low."""
    if SETTINGS.is_fake_llm():
        return base  # fake LLM ≡ rules
    try:
        prompt = (
            "你是电商客服 NLU 模块,请基于用户问题输出 JSON: "
            '{"intent": str, "intent_conf": float, "slots": object}.\n'
            f"候选 intent: product_search / product_detail / product_compare / "
            f"order_query / refund_request / faq_policy / memory_recall / smalltalk / unknown\n"
            f"用户问题: {text}"
        )
        llm = get_chat_model()
        msg = llm.invoke([HumanMessage(content=prompt)])
        content = getattr(msg, "content", "") or ""
        data = json.loads(content) if isinstance(content, str) else {}
        if data.get("intent"):
            base.update(data)
    except Exception:
        pass
    return base


@traced_node("nlu")
def nlu_node(state: AgentState) -> dict:
    text = state.get("user_input") or ""
    base = run_nlu(text)
    if base["intent_conf"] < LLM_FALLBACK_THRESHOLD or base["intent"] == "unknown":
        base = _maybe_llm_refine(text, base)

    # merge with previous slots from working memory if same intent topic
    existing_slots = dict(state.get("slots") or {})
    new_slots = base.get("slots") or {}
    merged = {**existing_slots, **new_slots}

    # working memory: keep last_intent + accumulated slots
    working = dict(state.get("memory_working") or {})
    working["last_intent"] = base["intent"]
    working["slots"] = merged

    return {
        "intent": base["intent"],
        "intent_conf": base["intent_conf"],
        "slots": merged,
        "memory_working": working,
    }
