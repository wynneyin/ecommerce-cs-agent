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


def _hot_product_browse_lines(n: int = 4) -> str:
    """无严匹配时给用户提供可点的热卖备选（正反馈）。"""
    from src.tools.data_loader import load_products

    hot = sorted(load_products(), key=lambda p: p.get("rating", 0), reverse=True)[:n]
    if not hot:
        return ""
    lines = "\n".join(
        f"- {p['name']}（`{p['product_id']}`，¥{p['price']}，{p['rating']} 分）"
        for p in hot
    )
    return "\n\n给您参考几款店里口碑不错的商品，您看看有没有接近的：\n" + lines


def _truncate_facts(obj: Any, n: int = 3200) -> str:
    s = json.dumps(obj, ensure_ascii=False, default=str)
    if len(s) > n:
        return s[:n] + "…"
    return s


_TOOL_TO_INTENT: dict[str, str] = {
    "web_search": "web_search",
    "search_products": "product_search",
    "get_product_detail": "product_detail",
    "compare_products": "product_compare",
    "query_order": "order_query",
    "refund_request": "refund_request",
    "faq_retrieve": "faq_policy",
}


def _has_web_in_actions(actions: list[Any]) -> bool:
    return any(a.get("name") == "web_search" for a in (actions or []) if isinstance(a, dict))


def _bundle_tool_facts_for_synthesis(actions: list[Any], intent: str) -> tuple[str, bool]:
    """多工具（如 web_search + search_products）合并进合成 prompt；返回 (facts, suggestive_browse)。"""
    has_web = _has_web_in_actions(actions)
    if not has_web and len(actions) <= 1:
        last = actions[-1] if actions else {}
        lo = last.get("output") if isinstance(last, dict) else None
        sug = intent == "product_search" and isinstance(lo, dict) and bool(lo.get("suggestive"))
        return (_truncate_facts(lo) if lo is not None else "(无)", sug)
    parts: list[str] = []
    suggestive = False
    for a in actions:
        if not isinstance(a, dict) or not a.get("ok"):
            continue
        o = a.get("output")
        if not isinstance(o, dict):
            continue
        n = a.get("name")
        if n == "web_search":
            slim = {
                "query": o.get("query"),
                "summary": o.get("summary"),
                "items": (o.get("items") or [])[:8],
            }
            parts.append("【联网检索】" + _truncate_facts(slim, n=2000))
        else:
            parts.append(f"【{n}】" + _truncate_facts(o))
        if n == "search_products" and o.get("suggestive"):
            suggestive = True
    return ("\n\n".join(parts) if parts else "(无)"), (suggestive or has_web)


def _render_chained_tool_outputs(actions: list[Any]) -> str | None:
    chunks: list[str] = []
    for a in actions:
        if not isinstance(a, dict) or not a.get("ok"):
            continue
        o = a.get("output")
        if not isinstance(o, dict):
            continue
        name = a.get("name")
        fk = _TOOL_TO_INTENT.get(name or "")
        if not fk:
            continue
        t = _from_action_output(fk, o)
        if t:
            chunks.append(t)
    return "\n\n".join(chunks) if chunks else None


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
    facts, suggestive = _bundle_tool_facts_for_synthesis(actions, intent)

    prompt = build_synthesis_prompt(
        user_input=user_input,
        intent=intent,
        facts=facts,
        thinking=thinking,
        suggestive_browse=suggestive,
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
        tier = str(out.get("match_tier") or "strong")
        suggestive = bool(out.get("suggestive"))
        if not items:
            return (
                "按您这句暂时没筛到严丝合缝的款，先给您看几款店里卖得好的作参考；"
                "您补一句预算、品牌或用途，我帮您再精准缩一圈～"
                + _hot_product_browse_lines(5)
            )
        lines = "\n".join(
            f"- {it['name']}（`{it['product_id']}`，¥{it['price']}，{it['rating']} 分）"
            for it in items[:5]
        )
        if tier in ("soft", "browse") or suggestive:
            return (
                "没有完全一致的严匹配，下面这些是和您描述比较接近或通过评分帮您挑的备选，可以先扫一眼；"
                "您愿意的话再说说预算、品牌或用途，我能帮您缩得更准～\n\n"
                + lines
            )
        return "为您找到以下商品：\n" + lines

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

    if intent == "web_search":
        if not out.get("ok"):
            return out.get("hint") or out.get("error") or "联网搜索暂时不可用，请稍后再试。"
        summary = (out.get("summary") or "").strip()
        if summary:
            return "网上检索到以下参考（摘要）：\n" + summary
        return "未检索到可用的网页摘要，您可以换个说法或关键词再试一次。"

    if intent == "faq_policy":
        items = out.get("items") or []
        if not items:
            return (
                "没匹配到单独的政策页，您可以换个说法（如运费、退换、发票）试试；"
                "或直接说说您遇到的订单情况，我按场景帮您说明。"
            )
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
        + _hot_product_browse_lines(3)
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

    ok_tools = [
        a
        for a in actions
        if isinstance(a, dict)
        and a.get("ok")
        and isinstance(a.get("output"), dict)
        and _TOOL_TO_INTENT.get(a.get("name") or "")
    ]
    rendered: str | None = None
    if len(ok_tools) > 1 or (_has_web_in_actions(actions) and len(ok_tools) >= 1):
        rendered = _render_chained_tool_outputs(actions)
    elif last and last.get("ok") and isinstance(last.get("output"), dict):
        name = last.get("name")
        fk = _TOOL_TO_INTENT.get(name or "")
        if fk:
            rendered = _from_action_output(fk, last["output"])
        else:
            rendered = _from_action_output(intent, last["output"])

    if rendered:
        if SETTINGS.use_llm_conversational_fallback() and not SETTINGS.is_fake_llm():
            conv, patch = _llm_conversational_reply(state, template_hint=rendered)
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
