"""Rule-based stand-in for an LLM.

The Fake LLM exposes a `.invoke(messages_or_str) -> AIMessage`-style API so it
can be plugged anywhere a real chat model would be used. Heuristics live in
`rules.py`; this module just dispatches.

The graph nodes never *require* an LLM (they fall back to deterministic logic
on `LLM_PROVIDER=fake`). The fake LLM is mainly used for:
* the optional NLU "fallback" path,
* a friendly final-response composer,
* keeping the architecture honest by exercising the same code path as a real
  chat-model call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

from src.llm.rules import compose_response, plan_for_intent, run_nlu


@dataclass
class FakeChatModel:
    """Minimal chat-model surface used by the project."""

    name: str = "fake-zh"

    def invoke(self, messages: Any, config: Any | None = None) -> AIMessage:  # noqa: ARG002
        text = _to_text(messages)
        # If the prompt looks like a JSON-instruction NLU prompt, return JSON;
        # otherwise generate a short customer-service answer.
        if "返回 JSON" in text or '"intent"' in text:
            nlu = run_nlu(_extract_user_query(text))
            return AIMessage(content=json.dumps(nlu, ensure_ascii=False))
        if "请规划" in text or '"plan"' in text:
            intent = _grab(text, "intent=", "\n") or "unknown"
            plan = plan_for_intent(intent, _safe_json(_grab(text, "slots=", "\n") or "{}"))
            return AIMessage(content=json.dumps(plan, ensure_ascii=False))
        # Default: customer-service style reply
        return AIMessage(content=compose_response(text))

    # Compatibility with langchain Runnable
    def __or__(self, other: Any) -> Any:  # pragma: no cover
        return other.__ror__(self) if hasattr(other, "__ror__") else other

    def with_structured_output(self, schema: Any):  # pragma: no cover
        del schema
        return self


def _to_text(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list):
        parts = []
        for m in messages:
            if isinstance(m, BaseMessage):
                parts.append(str(m.content))
            elif isinstance(m, dict):
                parts.append(str(m.get("content", "")))
            else:
                parts.append(str(m))
        return "\n".join(parts)
    return str(messages)


def _extract_user_query(text: str) -> str:
    marker = "用户问题:"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text


def _grab(text: str, start: str, end: str) -> str | None:
    if start not in text:
        return None
    rest = text.split(start, 1)[1]
    return rest.split(end, 1)[0].strip()


def _safe_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        return {}
