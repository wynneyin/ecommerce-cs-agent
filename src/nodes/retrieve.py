"""Retrieve node — runs hybrid retrieval depending on intent."""

from __future__ import annotations

from src.retrieval import get_retriever
from src.state import AgentState
from src.trace import traced_node


def _build_query(state: AgentState) -> tuple[str, str]:
    """Return (kind, query) — kind selects which retriever to use."""
    intent = state.get("intent")
    slots = state.get("slots") or {}
    text = state.get("user_input") or ""

    if intent == "faq_policy":
        return "faq", text
    if intent == "product_search":
        parts = [text]
        if slots.get("category"):
            parts.append(slots["category"])
        if slots.get("budget"):
            parts.append(f"{slots['budget']} 元")
        return "product", " ".join(parts)
    if intent == "product_detail" and slots.get("product_id"):
        return "product", slots["product_id"]
    return "faq", text


@traced_node("retrieve")
def retrieve_node(state: AgentState) -> dict:
    kind, query = _build_query(state)
    retriever = get_retriever(kind)
    docs = retriever.retrieve(query, top_k=5)
    return {
        "retrieved": docs,
        "retrieval_method": retriever.last_method,
        "retrieval_query": query,
    }
