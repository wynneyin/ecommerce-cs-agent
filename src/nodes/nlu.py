"""NLU node — rule baseline + optional full LLM NLU when provider is not fake."""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import HumanMessage

from src.config import SETTINGS
from src.llm import get_chat_model, run_nlu
from src.llm.io_trace import merge_llm_summary
from src.llm.json_utils import parse_json_object
from src.llm.rules import INTENTS
from src.state import AgentState
from src.trace import traced_node

LLM_FALLBACK_THRESHOLD = 0.5

VALID_INTENTS = frozenset(INTENTS)


def _maybe_llm_refine(state: AgentState, text: str, base: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Low-confidence rule NLU: single LLM JSON pass (when USE_LLM_NLU is off)."""
    patch: dict[str, Any] = {}
    if SETTINGS.is_fake_llm():
        return base, patch
    try:
        prompt = (
            "你是电商客服 NLU 模块,请基于用户问题只输出一个 JSON 对象: "
            '{"intent": str, "intent_conf": float, "slots": object}。\n'
            "intent 必须是: product_search / product_detail / product_compare / "
            "order_query / refund_request / faq_policy / web_search / memory_recall / smalltalk / unknown\n"
            f"用户问题: {text}"
        )
        llm = get_chat_model()
        msg = llm.invoke([HumanMessage(content=prompt)])
        content = getattr(msg, "content", "") or ""
        patch = merge_llm_summary(state, "nlu_refine", content, prompt_hint=prompt[:1200])
        data = parse_json_object(content)
        if data and data.get("intent"):
            base.update(data)
    except Exception as exc:
        patch = merge_llm_summary(
            state,
            "nlu_refine_error",
            repr(exc),
            prompt_hint="invoke failed",
        )
    return base, patch


def _llm_primary_nlu(state: AgentState, text: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Primary NLU via chat model (JSON). Returns (parsed | None, llm_summary_patch)."""
    prompt = (
        "你是电商意图识别模块。只输出一个 JSON 对象，不要 Markdown，不要解释文字。\n"
        '键: "intent" (字符串), "intent_conf" (0到1的小数), "slots" (对象)。\n'
        "intent 必须是下列之一:\n"
        "product_search, product_detail, product_compare, order_query, refund_request, "
        "faq_policy, web_search, memory_recall, smalltalk, unknown\n"
        "slots 尽量从原文抽取可用字段，不必填满；拿不准的不要编造，可省略键或使用空对象 {}。\n"
        "常见键示例: order_id, product_id, product_ids, category, budget, quantity, refund_reason, query。\n"
        f"用户原话: {text}"
    )
    llm = get_chat_model()
    msg = llm.invoke([HumanMessage(content=prompt)])
    content = getattr(msg, "content", "") or ""
    patch = merge_llm_summary(state, "nlu_primary", content, prompt_hint=prompt[:1200])
    data = parse_json_object(content)
    if not data or not data.get("intent"):
        return None, patch
    intent = str(data["intent"]).strip()
    if intent not in VALID_INTENTS:
        return None, patch
    try:
        conf = float(data.get("intent_conf", 0.85))
    except (TypeError, ValueError):
        conf = 0.85
    slots = data.get("slots") if isinstance(data.get("slots"), dict) else {}
    return {"intent": intent, "intent_conf": conf, "slots": slots}, patch


@traced_node("nlu")
def nlu_node(state: AgentState) -> dict:
    text = state.get("user_input") or ""
    timing: dict[str, float] = {}
    llm_patch: dict[str, Any] = {}

    t0 = time.perf_counter()
    rules = run_nlu(text)
    timing["rules_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    llm_ms_total = 0.0

    if rules["intent"] == "unsafe":
        base = rules
    elif SETTINGS.use_llm_nlu():
        t1 = time.perf_counter()
        llm_part, llm_patch = _llm_primary_nlu(state, text)
        llm_ms_total = (time.perf_counter() - t1) * 1000
        timing["llm_invoke_ms"] = round(llm_ms_total, 2)
        if llm_part:
            merged_slots = {**(llm_part.get("slots") or {}), **(rules.get("slots") or {})}
            base = {
                "intent": llm_part["intent"],
                "intent_conf": llm_part["intent_conf"],
                "slots": merged_slots,
            }
        else:
            base = rules
    else:
        base = rules
        if not SETTINGS.is_fake_llm() and (
            base["intent_conf"] < LLM_FALLBACK_THRESHOLD or base["intent"] == "unknown"
        ):
            t1 = time.perf_counter()
            base, llm_patch = _maybe_llm_refine(state, text, base)
            llm_ms_total = (time.perf_counter() - t1) * 1000
            timing["llm_invoke_ms"] = round(llm_ms_total, 2)

    if "llm_invoke_ms" not in timing:
        timing["llm_invoke_ms"] = 0.0

    existing_slots = dict(state.get("slots") or {})
    new_slots = base.get("slots") or {}
    merged = {**existing_slots, **new_slots}

    working = dict(state.get("memory_working") or {})
    working["last_intent"] = base["intent"]
    working["slots"] = merged

    out: dict[str, Any] = {
        "intent": base["intent"],
        "intent_conf": base["intent_conf"],
        "slots": merged,
        "memory_working": working,
        "nlu_timing_ms": timing,
    }
    out.update(llm_patch)
    return out
