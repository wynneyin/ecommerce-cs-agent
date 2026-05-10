"""Understand node — LLM decomposes user input before NLU (primary reasoning surface)."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.config import SETTINGS
from src.llm import get_chat_model
from src.llm.io_trace import merge_llm_summary
from src.llm.persona_prompts import build_understand_prompt
from src.state import AgentState
from src.trace import traced_node


@traced_node("understand")
def understand_node(state: AgentState) -> dict:
    text = (state.get("user_input") or "").strip()
    if not SETTINGS.use_llm_query_understanding():
        return {
            "query_understanding": (
                "（未启用 USE_LLM_UNDERSTAND 或非远程模型：跳过 LLM 拆解。）"
            )
        }

    prompt = build_understand_prompt(text)
    try:
        llm = get_chat_model()
        msg = llm.invoke([HumanMessage(content=prompt)])
        out = (getattr(msg, "content", "") or "").strip()
        if not out:
            return {"query_understanding": "（模型未返回拆解文本）"}
        patch = merge_llm_summary(
            state,
            "query_understanding",
            out,
            prompt_hint=prompt[:1200],
        )
        return {"query_understanding": out, **patch}
    except Exception as exc:
        patch = merge_llm_summary(
            state,
            "query_understanding_error",
            repr(exc),
            prompt_hint=prompt[:800],
        )
        return {
            "query_understanding": f"（拆解调用异常：{exc!r}）",
            **patch,
        }
