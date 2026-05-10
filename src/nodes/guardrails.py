"""Guardrails node — filters unsafe / out-of-scope inputs."""

from __future__ import annotations

from src.state import AgentState
from src.trace import traced_node

UNSAFE_KEYWORDS = (
    "炸药",
    "毒品",
    "色情",
    "枪支",
    "破解",
    "密码盗",
    "信用卡盗刷",
    "黑客",
)
MAX_LEN = 500


@traced_node("guardrails")
def guardrails_node(state: AgentState) -> dict:
    text = (state.get("user_input") or "").strip()
    if not text:
        return {
            "guardrails_pass": False,
            "guardrails_reason": "empty_input",
            "final_response": "请告诉我您需要的帮助~",
        }
    if len(text) > MAX_LEN:
        return {
            "guardrails_pass": False,
            "guardrails_reason": "too_long",
            "final_response": "您的问题过长,请精简后再发送(<=500 字)。",
        }
    for kw in UNSAFE_KEYWORDS:
        if kw in text:
            return {
                "guardrails_pass": False,
                "guardrails_reason": f"unsafe:{kw}",
                "final_response": "抱歉,该话题不在客服支持范围内。如需帮助请提供电商相关问题。",
            }
    return {"guardrails_pass": True, "guardrails_reason": None}
