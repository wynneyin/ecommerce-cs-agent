"""Memory update node — promotes useful facts from working → long-term and
composes the final, user-facing response."""

from __future__ import annotations

from src.memory import LongTermMemory
from src.state import AgentState
from src.trace import traced_node


_LONG_MEM = LongTermMemory()


def _from_action_output(intent: str, out: dict) -> str | None:
    """Render a final response from the last successful tool output."""
    if intent == "product_search":
        items = out.get("items") or []
        if not items:
            return "没有找到匹配的商品,可以换个关键词或放宽预算试试~"
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
        # Resolve the *original* FAQ body via topic id (the indexed text contains
        # keyword soup, which is bad for direct presentation).
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
        return "请提供您要退款的订单号(例如 E202603000001),并简单说明原因。"
    if intent == "refund_request":
        return "退款流程暂未走通,请检查订单号是否正确,或换种描述。"
    if intent == "order_query" and not slots.get("order_id"):
        return "请提供您要查询的订单号(例如 E202603000001)。"
    if intent == "order_query":
        return "未能查到该订单,请检查订单号是否正确。"

    if intent == "memory_recall":
        mem = state.get("memory_long") or {}
        if not mem:
            return "暂无相关记忆,可以告诉我您想做什么吗?"
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
            return "我记得您之前关注的是 " + "、".join(parts) + "。需要继续上次的话题吗?"
        recall = mem.get("last_intent")
        if recall:
            return f"我记得您上一次咨询的是:{recall}。需要继续吗?"
        return "暂无相关记忆,可以告诉我您想做什么吗?"

    if intent == "smalltalk":
        return "您好,我是客服助手,请问有什么可以帮您?"
    if intent == "unsafe":
        return "抱歉,该话题不在客服支持范围内。"

    if state.get("guardrails_reason"):
        return "抱歉,我无法处理该请求。"

    return "暂未理解您的问题,可以尝试更具体地描述,例如「订单 E202603000001 物流到哪了?」"


def _final_response_from_state(state: AgentState) -> str:
    if state.get("final_response"):
        return state["final_response"]

    intent = state.get("intent") or "unknown"
    actions = state.get("actions") or []
    last = actions[-1] if actions else None

    if last and last.get("ok") and isinstance(last.get("output"), dict):
        rendered = _from_action_output(intent, last["output"])
        if rendered:
            return rendered

    return _fallback_response(state)


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

    return {
        "memory_long": long_mem,
        "final_response": _final_response_from_state(state),
    }
