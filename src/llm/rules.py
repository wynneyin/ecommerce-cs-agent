"""Chinese rule-based NLU + planning + response composition.

These rules power the *fake* LLM but are also the deterministic NLU baseline
used by the real graph (LLM is only a fallback). Putting them in one place
keeps behaviour reproducible and evaluable.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Intent labels (kept in sync with the eval dataset taxonomy)
# ---------------------------------------------------------------------------

INTENTS = [
    "product_search",
    "product_detail",
    "product_compare",
    "order_query",
    "refund_request",
    "faq_policy",
    "memory_recall",
    "smalltalk",
    "unsafe",
    "unknown",
]


# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

REFUND_KEYWORDS = (
    "退款",
    "退货",
    "退掉",
    "退一下",
    "申请退",
    "售后",
    "想退",
    "要退",
    "退了",
    "退吧",
    "这单退",
    "把这单退",
)
COMPARE_KEYWORDS = ("对比", "比较", "哪个好", "区别", "差别", "vs", "VS")
DETAIL_KEYWORDS = ("详情", "参数", "规格", "介绍一下", "怎么样", "是什么", "了解一下")
SEARCH_KEYWORDS = ("推荐", "想买", "找", "搜", "有什么", "适合", "性价比", "预算")
ORDER_KEYWORDS = ("订单", "发货", "物流", "快递", "到哪", "什么时候到", "我的单")
FAQ_KEYWORDS = (
    "政策",
    "运费",
    "包邮",
    "几天",
    "多久",
    "保修",
    "三包",
    "发票",
    "开票",
    "增值税",
    "无理由",
    "七天",
    "会员",
    "积分",
    "等级",
    "优惠券",
    "满减",
    "怎么联系",
    "联系客服",
    "客服电话",
    "投诉",
    "可以退吗",
    "支付",
    "支付方式",
    "怎么付",
    "花呗",
    "分期",
    "白条",
    "改地址",
    "修改地址",
    "修改收货",
    "收货地址",
    "换货",
    "换码",
    "换尺码",
    "换颜色",
    "价保",
    "保价",
    "缺货",
    "补货",
)
FAQ_HOW_PREFIXES = (
    "怎么",
    "如何",
    "可以",
    "支持吗",
    "支不支持",
    "可不可以",
    "什么时候",
    "几天",
    "多久",
)
MEMORY_KEYWORDS = ("刚才", "之前", "上次", "记得", "我说过", "刚说", "再来一次", "还是那个")
UNSAFE_PATTERNS = (
    "炸药",
    "色情",
    "毒品",
    "枪支",
    "破解密码",
    "信用卡盗刷",
)


ORDER_RE = re.compile(r"(?:订单(?:号)?[:： ]?)?(\b[A-Z]{1,3}\d{6,12}\b|\b\d{8,14}\b)")
SKU_RE = re.compile(r"\b(P\d{3,5}|SKU[-_]?\d{3,6})\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# NLU
# ---------------------------------------------------------------------------


def run_nlu(text: str) -> dict[str, Any]:
    """Heuristic Chinese NLU returning intent + slots + confidence."""
    raw = (text or "").strip()
    low = raw.lower()
    slots: dict[str, Any] = {}

    # 1. unsafe first
    if any(p in raw for p in UNSAFE_PATTERNS):
        return {"intent": "unsafe", "intent_conf": 0.99, "slots": {}}

    # 2. slots: order id / sku / quantity / budget
    order_match = ORDER_RE.search(raw)
    if order_match:
        slots["order_id"] = order_match.group(1)

    sku_matches = SKU_RE.findall(raw)
    if sku_matches:
        skus = [s.upper().replace("SKU-", "P").replace("SKU_", "P").replace("SKU", "P") for s in sku_matches]
        if len(skus) == 1:
            slots["product_id"] = skus[0]
        else:
            slots["product_ids"] = skus

    # Budget: catch "5000 元", "5000 内", "预算 5000", "5000 以内" and similar
    money = re.search(
        r"(\d{2,5})\s*(?:元|块|rmb|￥|内|以内)|预算[:： ]*(\d{2,5})",
        raw,
        re.IGNORECASE,
    )
    if money:
        slots["budget"] = int(money.group(1) or money.group(2))

    qty = re.search(r"(\d+)\s*(?:件|个|台|条|双)", raw)
    if qty:
        slots["quantity"] = int(qty.group(1))

    # 3. category / topic hint
    for cat in ("手机", "笔记本", "电脑", "耳机", "音箱", "相机", "平板", "手表", "鞋", "服装", "面膜"):
        if cat in raw:
            slots["category"] = cat
            break

    # 4. memory recall (must precede generic search)
    if any(k in raw for k in MEMORY_KEYWORDS):
        intent = "memory_recall"
        conf = 0.85
        return {"intent": intent, "intent_conf": conf, "slots": slots}

    has_order_id = "order_id" in slots
    is_how_question = any(p in raw for p in FAQ_HOW_PREFIXES)
    matches_faq = any(k in raw for k in FAQ_KEYWORDS)

    # 5. refund vs FAQ disambiguation:
    #    "七天无理由退货怎么操作" / "退款政策" → FAQ (no order_id, asking how)
    #    "订单 X 想退款" / "退款 X" → refund_request (operational)
    if any(k in raw for k in REFUND_KEYWORDS):
        is_how_refund = is_how_question or any(
            k in raw for k in ("政策", "几天", "多久", "条件", "可以吗", "七天", "无理由")
        )
        if is_how_refund and not has_order_id:
            return {"intent": "faq_policy", "intent_conf": 0.92, "slots": slots}

        reason = _extract_refund_reason(raw)
        if reason:
            slots["refund_reason"] = reason
        return {"intent": "refund_request", "intent_conf": 0.95, "slots": slots}

    # 6. compare
    if any(k in raw for k in COMPARE_KEYWORDS) or (slots.get("product_ids") and len(slots["product_ids"]) >= 2):
        return {"intent": "product_compare", "intent_conf": 0.9, "slots": slots}

    # 7. order vs FAQ disambiguation:
    #    "几天发货" / "怎么取消订单" → FAQ (how-questions without order_id)
    #    "订单 X 物流到哪了" → order_query
    if has_order_id:
        return {"intent": "order_query", "intent_conf": 0.93, "slots": slots}
    if any(k in raw for k in ORDER_KEYWORDS):
        if is_how_question or matches_faq:
            return {"intent": "faq_policy", "intent_conf": 0.9, "slots": slots}
        return {"intent": "order_query", "intent_conf": 0.85, "slots": slots}

    # 8. detail (single product mention or detail keywords)
    if "product_id" in slots or any(k in raw for k in DETAIL_KEYWORDS):
        return {"intent": "product_detail", "intent_conf": 0.85, "slots": slots}

    # 9. FAQ / policy
    if matches_faq or is_how_question or raw.endswith("吗?") or raw.endswith("吗?"):
        return {"intent": "faq_policy", "intent_conf": 0.85, "slots": slots}

    # 10. search
    if any(k in raw for k in SEARCH_KEYWORDS) or "category" in slots:
        return {"intent": "product_search", "intent_conf": 0.75, "slots": slots}

    # 11. smalltalk
    if any(g in raw for g in ("你好", "hi", "hello", "在吗", "谢谢")):
        return {"intent": "smalltalk", "intent_conf": 0.7, "slots": slots}

    return {"intent": "unknown", "intent_conf": 0.3, "slots": slots}


def _extract_refund_reason(text: str) -> str | None:
    pairs = [
        ("质量", "质量问题"),
        ("坏了", "质量问题"),
        ("破损", "质量问题"),
        ("尺码不对", "尺码不合"),
        ("尺码", "尺码不合"),
        ("不喜欢", "不喜欢"),
        ("不想要", "不想要"),
        ("发错", "发货错误"),
        ("漏发", "漏发"),
        ("假货", "疑似假货"),
        ("太贵", "降价补差"),
    ]
    for k, v in pairs:
        if k in text:
            return v
    return None


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def plan_for_intent(intent: str, slots: dict[str, Any]) -> dict[str, Any]:
    """Map (intent, slots) to a canonical tool plan."""
    plan: list[dict[str, Any]] = []
    if intent == "product_search":
        plan.append(
            {
                "name": "search_products",
                "args": {
                    "query": slots.get("query", ""),
                    "category": slots.get("category"),
                    "budget": slots.get("budget"),
                },
            }
        )
    elif intent == "product_detail":
        plan.append({"name": "get_product_detail", "args": {"product_id": slots.get("product_id")}})
    elif intent == "product_compare":
        plan.append({"name": "compare_products", "args": {"product_ids": slots.get("product_ids", [])}})
    elif intent == "order_query":
        plan.append({"name": "query_order", "args": {"order_id": slots.get("order_id")}})
    elif intent == "refund_request":
        plan.append(
            {
                "name": "refund_request",
                "args": {
                    "order_id": slots.get("order_id"),
                    "reason": slots.get("refund_reason", "未填写"),
                },
            }
        )
    elif intent == "faq_policy":
        plan.append({"name": "faq_retrieve", "args": {"query": slots.get("query", "")}})
    elif intent == "memory_recall":
        plan.append({"name": "memory_lookup", "args": {}})
    return {"plan": plan}


# ---------------------------------------------------------------------------
# Response composer
# ---------------------------------------------------------------------------


def compose_response(text: str) -> str:
    """Generate a friendly customer-service style reply.

    The deterministic graph builds the *content* by formatting tool outputs;
    this fallback is only used when nodes ask the LLM for free-form text.
    """
    return "已为您处理上述请求,如还有其他问题请随时告知~"
