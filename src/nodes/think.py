"""Thinking step — LLM reasoning between retrieval observation and planning."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.config import SETTINGS
from src.llm import get_chat_model
from src.llm.io_trace import merge_llm_summary
from src.state import AgentState
from src.trace import traced_node


def _summarize_retrieved(state: AgentState) -> str:
    docs = state.get("retrieved") or []
    if not docs:
        return "(本轮未检索到文档)"
    lines: list[str] = []
    for d in docs[:6]:
        tid = d.get("id", "")
        title = (d.get("metadata") or {}).get("title") or tid
        sc = d.get("score", 0)
        lines.append(f"- [{tid}] {title} (score≈{sc})")
    return "\n".join(lines)


@traced_node("think")
def think_node(state: AgentState) -> dict:
    if not SETTINGS.use_llm_thinking():
        return {"thinking": ""}

    user = state.get("user_input") or ""
    intent = state.get("intent") or "unknown"
    slots = state.get("slots") or {}
    obs = (state.get("memory_working") or {}).get("last_observation") or {}
    react_prior = state.get("react_trace") or []
    replan_hint = ""
    if react_prior:
        last_r = react_prior[-1]
        replan_hint = (
            "\n【重规划上下文】上一轮反思结论（机械）："
            f"need_replan={last_r.get('need_replan')}, code={last_r.get('reflection_code')}。\n"
            f"文字反思摘要：{(last_r.get('react_reasoning') or '')[:900]}\n"
            "请在本轮思考中明确：为何要调整策略、下一步计划侧重点。\n"
        )

    prompt = (
        "你是电商客服系统的内部推理模块（Think，发生在 Plan 之前）。根据下方上下文，用简洁中文写出你的思考过程"
        "（3～8 句）：①用户想要什么；②已有槽位与检索线索；③下一步应执行的客服策略。\n"
        "不要输出 JSON 或代码块，不要使用 Markdown 标题符号。\n\n"
        f"用户原话：{user}\n"
        f"NLU 意图：{intent}\n"
        f"槽位：{slots}\n"
        f"检索摘要：{_summarize_retrieved(state)}\n"
        f"观察：{obs}\n"
        f"{replan_hint}"
    )
    try:
        llm = get_chat_model()
        msg = llm.invoke([HumanMessage(content=prompt)])
        text = getattr(msg, "content", "") or ""
        text = str(text).strip()
        working = dict(state.get("memory_working") or {})
        rounds = list(working.get("think_rounds") or [])
        rounds.append(text)
        working["think_rounds"] = rounds
        trace_patch = merge_llm_summary(state, "think", text, prompt_hint=prompt[:1200])
        return {"thinking": text, "memory_working": working, **trace_patch}
    except Exception as exc:
        err_patch = merge_llm_summary(
            state,
            "think_error",
            repr(exc),
            prompt_hint=prompt[:800],
        )
        return {"thinking": "", **err_patch}
