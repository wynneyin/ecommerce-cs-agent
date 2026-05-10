"""LLM 决定是否在本轮编排中前置调用 ``web_search``（不限于用户说「联网搜索」）。"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from src.config import SETTINGS
from src.llm import get_chat_model
from src.llm.json_utils import parse_json_object
from src.state import AgentState


def augment_plan_with_web_search(state: AgentState, plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """在已有计划前插入 ``web_search``（若模型判断需要联网补充）。"""
    if not SETTINGS.use_llm_web_tool():
        return plan
    intent = state.get("intent") or "unknown"
    if intent == "web_search":
        return plan
    if intent in {"unsafe", "refund_request", "order_query", "memory_recall"}:
        return plan
    if plan and plan[0].get("name") == "web_search":
        return plan

    user = (state.get("user_input") or "").strip()
    if not user:
        return plan

    slots = state.get("slots") or {}
    thinking = (state.get("thinking") or "").strip()[:900]
    planned_names = [p.get("name") for p in plan if p.get("name")]

    prompt = (
        "你是电商客服编排器，只判断本轮是否需要调用联网搜索工具 web_search。\n"
        "需要联网的典型情况：实时行情/新品发售、第三方评测、品牌官网参数、竞品对比、"
        "明显超出店铺知识库的客观事实、用户明确要求网上查。\n"
        "不需要联网：仅查本店订单/退款/物流单号、仅用店内 FAQ/商品库即可回答、纯寒暄。\n\n"
        "只输出一个 JSON 对象，不要 Markdown，不要解释：\n"
        '{"need_web": true 或 false, "search_query": "若 need_web 为 true 则填写给搜索引擎的简短中文查询（否则填空字符串）"}\n\n'
        f"用户原话：{user}\n"
        f"NLU 意图：{intent}\n"
        f"槽位：{slots}\n"
        f"已有工具计划（将先于联网执行或并行考虑）：{planned_names}\n"
        f"Think 摘要：{thinking or '（无）'}\n"
    )
    try:
        llm = get_chat_model()
        msg = llm.invoke([HumanMessage(content=prompt)])
        raw = getattr(msg, "content", "") or ""
        data = parse_json_object(str(raw))
        if not data:
            return plan
        need = bool(data.get("need_web"))
        q = str(data.get("search_query") or "").strip()
        if not need or not q:
            return plan
        web_call: dict[str, Any] = {"name": "web_search", "args": {"query": q}}
        return [web_call, *plan]
    except Exception:
        return plan
