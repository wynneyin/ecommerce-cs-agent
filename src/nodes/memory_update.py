"""Memory update node — promotes useful facts from working → long-term and
composes the final, user-facing response."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage

from src.config import SETTINGS
from src.llm import get_chat_model
from src.llm.io_trace import merge_llm_summary
from src.llm.persona_prompts import (
    build_conversational_fallback_prompt,
    build_synthesis_prompt,
)
from src.memory import LongTermMemory
from src.state import AgentState
from src.trace import traced_node


_LONG_MEM = LongTermMemory()


def _truncate_facts(obj: Any, n: int = 3200) -> str:
    s = json.dumps(obj, ensure_ascii=False, default=str)
    if len(s) > n:
        return s[:n] + "…"
    return s


def _llm_synthesize_final(state: AgentState) -> tuple[str | None, dict[str, Any]]:
    """Natural-language reply using tool outputs + persona/safety prompts."""
    if not SETTINGS.use_llm_synthesis():
        return None, {}
    actions = state.get("actions") or []
    if not actions:
        return None, {}

    user_input = state.get("user_input") or ""
    intent = state.get("intent") or "unknown"
    thinking = (state.get("thinking") or "").strip()
    last = actions[-1] if actions else None
    last_out = last.get("output") if isinstance(last, dict) else None
    facts = _truncate_facts(last_out) if last_out is not None else "(本轮无工具 JSON 输出)"

    prompt = build_synthesis_prompt(
        user_input=user_input,
        intent=intent,
        facts=facts,
        thinking=thinking,
    )
    try:
        llm = get_chat_model()
        msg = llm.invoke([HumanMessage(content=prompt)])
        out = (getattr(msg, "content", "") or "").strip()
        if not out:
            return None, {}
        patch = merge_llm_summary(
            state, "reply_synthesis", out, prompt_hint=prompt[:1200]
        )
        return out, patch
    except Exception as exc:
        patch = merge_llm_summary(
            state,
            "reply_synthesis_error",
            repr(exc),
            prompt_hint=prompt[:800],
        )
        return None, patch


def _llm_conversational_reply(
    state: AgentState, *, template_hint: str
) -> tuple[str | None, dict[str, Any]]:
    """Turn rigid template lines into warmer prose; keeps draft as factual anchor."""
    if not SETTINGS.use_llm_conversational_fallback():
        return None, {}
    intent = state.get("intent") or "unknown"
    if intent == "unsafe":
        return None, {}

    slots = state.get("slots") or {}
    slots_hint = json.dumps(slots, ensure_ascii=False) if slots else "(空)"
    mem = state.get("memory_long") or {}
    memory_hint = json.dumps(mem, ensure_ascii=False)[:800] if mem else ""

    extra = ""
    if template_hint.strip():
        extra = (
            "【以下为系统根据规则生成的要点草稿，请改写得更自然、像真人客服；"
            "不得编造草稿里没有的事实】\n"
            + template_hint.strip()[:2000]
        )

    prompt = build_conversational_fallback_prompt(
        user_input=state.get("user_input") or "",
        intent=intent,
        slots_hint=slots_hint,
        memory_hint=memory_hint,
        extra_context=extra,
    )
    try:
        llm = get_chat_model()
        msg = llm.invoke([HumanMessage(content=prompt)])
        out = (getattr(msg, "content", "") or "").strip()
        if not out:
            return None, {}
        patch = merge_llm_summary(
            state,
            "reply_conversational",
            out,
            prompt_hint=prompt[:1200],
        )
        return out, patch
    except Exception as exc:
        patch = merge_llm_summary(
            state,
            "reply_conversational_error",
            repr(exc),
            prompt_hint=prompt[:600],
        )
        return None, patch


def _from_action_output(intent: str, out: dict) -> str | None:
    """Render a final response from the last successful tool output."""
    if intent == "product_search":
        items = out.get("items") or []
        if not items:
            return "这边筛了一圈暂时没有命中商品，要不您换个关键词或说说预算区间，我再帮您找找？"
        return "为您找到以下商品:\n" + "\n".join(
            f"- {it['name']}({it['product_id']}, {it['price']} 元, {it['rating']} 分)"
            for it in items[:5]
        )

    if intent == "product_detail":
        it = out.get("item") or {}
        return (
            f"{it.get('name', '商品')}({it.get('product_id', '')})"
            f",售价 {it.get('price', '?')} 元,评分 {it.get('rating', '?')}。"
            f"\n卖点: {', '.join(it.get('tags', []))}"
            f"\n参数: {', '.join(map(str, it.get('specs', [])))}"
        )

    if intent == "product_compare":
        return out.get("summary") or "已为您完成对比。"

    if intent == "order_query":
        order = out.get("order") or {}
        if not order:
            return None
        base = (
            f"订单 {order.get('order_id', '')}: {order.get('product_name', '')} x"
            f"{order.get('quantity', 1)},金额 {order.get('amount', '?')} 元,"
            f"状态 {order.get('status_label', order.get('status', '未知'))}"
        )
        if order.get("courier") and order.get("tracking_no"):
            base += f",{order['courier']} 单号 {order['tracking_no']}"
        return base

    if intent == "refund_request":
        return out.get("message") or "退款申请已受理。"

    if intent == "faq_policy":
        items = out.get("items") or []
        if not items:
            return "暂未找到相关政策,请补充更多细节。"
        first = items[0]
        topic = first.get("metadata", {}).get("title") or first.get("id")
        from src.tools.data_loader import load_faq

        raw_body = ""
        for doc in load_faq():
            if doc["topic"] == first.get("id"):
                raw_body = doc["content"]
                break
        if not raw_body:
            raw_body = first.get("content") or ""
        body_lines = [
            ln.strip()
            for ln in raw_body.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        teaser = " ".join(body_lines[:2])[:240]
        return f"关于「{topic}」: {teaser}"

    return None


def _fallback_response(state: AgentState) -> str:
    intent = state.get("intent") or "unknown"
    slots = state.get("slots") or {}

    if intent == "refund_request" and not slots.get("order_id"):
        return "要帮您处理退款的话，需要订单号哦～发一下类似 E202603000001 这种号，并简单说下原因，这边帮您跟进。"
    if intent == "refund_request":
        return "退款这边暂时没查到可用记录，麻烦您核对订单号是否复制完整，或换个说法我再试试。"
    if intent == "order_query" and not slots.get("order_id"):
        return "查订单需要订单号～您发我一行类似 E202603000001 的号码就可以。"
    if intent == "order_query":
        return "这个订单号目前没查到结果，您方便确认一下是否输错，或换订单再试？"

    if intent == "memory_recall":
        mem = state.get("memory_long") or {}
        if not mem:
            return "我这边还没记下您刚才聊的细节，可以直接再说一下您想找什么吗？"
        parts: list[str] = []
        if mem.get("last_category"):
            parts.append(f"类目「{mem['last_category']}」")
        if mem.get("last_budget"):
            parts.append(f"预算 {mem['last_budget']} 元")
        if mem.get("last_product_id"):
            parts.append(f"商品 {mem['last_product_id']}")
        if mem.get("last_product_ids"):
            parts.append(f"商品 {' / '.join(mem['last_product_ids'])}")
        if mem.get("last_order_id"):
            parts.append(f"订单 {mem['last_order_id']}")
        if parts:
            return "我记得您刚才关注的是 " + "、".join(parts) + "。咱们继续从这个往下聊可以吗？"
        recall = mem.get("last_intent")
        if recall:
            return f"您上一条像是在问「{recall}」相关的事，还要接着聊这块吗？"
        return "我这边暂时没对上之前的上下文，您再用一句话说说需求就行～"

    if intent == "smalltalk":
        return "嗨～我是店铺客服，购物、订单、售后都可以问我，您想先从哪块开始？"
    if intent == "unsafe":
        return "抱歉，这类内容不在咱们客服受理范围内；若有购物相关问题可以随时问我。"

    if state.get("guardrails_reason"):
        return "抱歉，这条我这边没法处理；换成订单、商品或售后相关的问题我可以帮您看。"

    return (
        "这句我还没完全对上您的具体需求～可以说说订单号、商品类型或者遇到的现象，我帮您一步步看。"
    )


def _compose_final_response_bundle(
    state: AgentState, *, synthesized: str | None = None
) -> tuple[str, dict[str, Any]]:
    if state.get("final_response"):
        return state["final_response"], {}
    if synthesized:
        return synthesized, {}

    intent = state.get("intent") or "unknown"
    actions = state.get("actions") or []
    last = actions[-1] if actions else None

    if last and last.get("ok") and isinstance(last.get("output"), dict):
        rendered = _from_action_output(intent, last["output"])
        if rendered:
            if SETTINGS.use_llm_conversational_fallback() and not SETTINGS.is_fake_llm():
                conv, patch = _llm_conversational_reply(
                    state, template_hint=rendered
                )
                if conv:
                    return conv, patch
            return rendered, {}

    draft = _fallback_response(state)
    conv, patch = _llm_conversational_reply(state, template_hint=draft)
    if conv:
        return conv, patch
    return draft, {}


@traced_node("memory_update")
def memory_update_node(state: AgentState) -> dict:
    intent = state.get("intent")
    slots = state.get("slots") or {}
    user_id = state.get("user_id") or "anon"

    long_mem = dict(state.get("memory_long") or {})
    if intent and intent not in {"unknown", "unsafe", "smalltalk"}:
        long_mem["last_intent"] = intent
    if slots.get("category"):
        long_mem["last_category"] = slots["category"]
    if slots.get("budget"):
        long_mem["last_budget"] = slots["budget"]
    if slots.get("product_id"):
        long_mem["last_product_id"] = slots["product_id"]
    if slots.get("product_ids"):
        long_mem["last_product_ids"] = slots["product_ids"]
    if slots.get("order_id"):
        long_mem["last_order_id"] = slots["order_id"]

    _LONG_MEM.put(user_id, long_mem)

    syn, syn_patch = _llm_synthesize_final(state)
    fr, extra_patch = _compose_final_response_bundle(state, synthesized=syn)
    out: dict[str, Any] = {
        "memory_long": long_mem,
        "final_response": fr,
    }
    out.update(syn_patch)
    out.update(extra_patch)
    return out
