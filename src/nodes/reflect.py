"""Reflect node — checks whether actions satisfied the request and records ReAct trace.

When ``need_replan`` is True (react graph only), the router loops back to plan.
Optional LLM narrative explains the reflection step for UI observability.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage

from src.config import SETTINGS
from src.llm import get_chat_model
from src.llm.io_trace import merge_llm_summary
from src.state import AgentState
from src.trace import traced_node


MAX_REPLAN = 1


def _truncate_blob(obj: Any, limit: int = 1400) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    if len(s) > limit:
        return s[:limit] + "…"
    return s


def _llm_react_reflect_narrative(
    state: AgentState, *, mechanical: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    if not SETTINGS.use_llm_react_reflect():
        return "", {}
    user = state.get("user_input") or ""
    intent = state.get("intent") or ""
    actions = state.get("actions") or []
    last = actions[-1] if actions else {}
    name = last.get("name") if isinstance(last, dict) else None
    ok = last.get("ok") if isinstance(last, dict) else None
    out = last.get("output") if isinstance(last, dict) else None

    prompt = (
        "你是电商客服 ReAct 代理里的「反思(Reflect)」步骤。\n"
        "请用简洁中文写一小段（约 4～10 句），说明：\n"
        "① 用户想要什么；② 刚执行的工具是否满足需求、关键事实是什么；③"
        "若下方「机械判定」显示需要重规划，说明为何以及如何调整思路；④ 否则说明可以收尾。\n"
        "不要使用 Markdown 标题或代码围栏。\n\n"
        f"用户问题：{user}\n"
        f"意图：{intent}\n"
        f"机械判定（代码）：{mechanical}\n"
        f"最后工具：{name}  ·  ok={ok}\n"
        f"工具输出摘要：{_truncate_blob(out)}\n"
    )
    try:
        llm = get_chat_model()
        msg = llm.invoke([HumanMessage(content=prompt)])
        text = (getattr(msg, "content", "") or "").strip()
        patch = merge_llm_summary(state, "reflect", text, prompt_hint=prompt[:1200])
        return text, patch
    except Exception as exc:
        patch = merge_llm_summary(
            state,
            "reflect_error",
            repr(exc),
            prompt_hint=prompt[:800],
        )
        return "", patch


@traced_node("reflect")
def reflect_node(state: AgentState) -> dict:
    actions = state.get("actions") or []
    intent = state.get("intent")
    slots = state.get("slots") or {}
    replan_count = int(state.get("replan_count") or 0)

    last_ok = bool(actions and actions[-1].get("ok"))

    need_replan = False
    reason = "ok" if last_ok else "last_action_failed"

    if intent == "product_compare":
        last = actions[-1].get("output") if actions else None
        if not last or not (isinstance(last, dict) and last.get("items") and len(last["items"]) >= 2):
            if replan_count < MAX_REPLAN and len(slots.get("product_ids") or []) >= 2:
                need_replan = True
                reason = "compare_insufficient_items"

    if intent == "order_query":
        last = actions[-1].get("output") if actions else None
        if not last or not (isinstance(last, dict) and last.get("ok")):
            if replan_count < MAX_REPLAN:
                reason = "order_not_found"

    if need_replan:
        replan_count += 1

    mechanical = {
        "reflection_code": reason,
        "need_replan": need_replan,
        "replan_count": replan_count,
        "last_tool": actions[-1].get("name") if actions else None,
        "last_ok": last_ok,
    }

    narrative, ref_patch = _llm_react_reflect_narrative(state, mechanical=mechanical)

    prev = list(state.get("react_trace") or [])
    step: dict[str, Any] = {
        "round": len(prev) + 1,
        **mechanical,
    }
    if narrative:
        step["react_reasoning"] = narrative
    prev.append(step)

    out: dict[str, Any] = {
        "reflection": reason,
        "need_replan": need_replan,
        "replan_count": replan_count,
        "react_trace": prev,
    }
    out.update(ref_patch)
    return out
