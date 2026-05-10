"""Observe node — turn raw retrieval into a structured observation."""

from __future__ import annotations

from src.state import AgentState
from src.trace import traced_node


@traced_node("observe")
def observe_node(state: AgentState) -> dict:
    docs = state.get("retrieved") or []
    obs = {
        "num_docs": len(docs),
        "top_ids": [d.get("id") for d in docs[:5]],
        "top_topics": [d.get("metadata", {}).get("topic") for d in docs[:5]],
        "method": state.get("retrieval_method"),
    }
    working = dict(state.get("memory_working") or {})
    working["last_observation"] = obs
    return {"memory_working": working}
