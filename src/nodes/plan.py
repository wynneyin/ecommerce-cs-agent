"""Plan node — produces the next-step tool plan.

Behaviour by mode:
* ``deterministic`` — direct mapping (intent + slots) -> single ToolCall.
* ``react`` — same baseline plan; if reflection requested replan, the slot
  inputs may have been edited by the reflect node.
"""

from __future__ import annotations

from typing import Any

from src.llm import plan_for_intent
from src.llm.web_plan import augment_plan_with_web_search
from src.state import AgentState
from src.trace import traced_node


def _retrieved_product_seed_ids(retrieved: list[Any]) -> list[str]:
    """IDs from hybrid retrieve (BM25+向量+RRF)，供 search_products 与关键词结果融合。"""
    out: list[str] = []
    seen: set[str] = set()
    for doc in retrieved:
        meta = doc.get("metadata") or {}
        if meta.get("source") != "product":
            continue
        pid = doc.get("id")
        if isinstance(pid, str) and pid and pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


@traced_node("plan")
def plan_node(state: AgentState) -> dict:
    intent = state.get("intent") or "unknown"
    slots = dict(state.get("slots") or {})
    # Inject query for searches / FAQ / web
    if intent in {"product_search", "faq_policy", "web_search"}:
        slots.setdefault("query", state.get("user_input") or "")
    # For memory_recall, plan stays empty; final response comes from memory
    plan_obj = plan_for_intent(intent, slots)
    plan: list[dict] = list(plan_obj.get("plan", []))
    if intent == "product_search" and plan:
        tc = plan[0]
        if tc.get("name") == "search_products":
            seeds = _retrieved_product_seed_ids(state.get("retrieved") or [])
            if seeds:
                args = dict(tc.get("args") or {})
                args["seed_product_ids"] = seeds
                plan[0] = {**tc, "args": args}

    plan = augment_plan_with_web_search(state, plan)
    return {"plan": plan}
