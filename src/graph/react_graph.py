"""ReAct graph — same nodes, but with a plan↔reflect loop (bounded by reflect)."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.graph.router import after_act, after_guardrails, after_nlu, after_reflect
from src.memory import get_checkpointer
from src.nodes import (
    act_node,
    confirm_node,
    guardrails_node,
    memory_update_node,
    nlu_node,
    observe_node,
    plan_node,
    reflect_node,
    retrieve_node,
    think_node,
)
from src.state import AgentState


def build_react_graph(*, with_checkpointer: bool = True):
    g = StateGraph(AgentState)

    g.add_node("guardrails", guardrails_node)
    g.add_node("nlu", nlu_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("observe", observe_node)
    g.add_node("think", think_node)
    g.add_node("plan", plan_node)
    g.add_node("act", act_node)
    g.add_node("confirm", confirm_node)
    g.add_node("reflect", reflect_node)
    g.add_node("memory_update", memory_update_node)

    g.add_edge(START, "guardrails")
    g.add_conditional_edges(
        "guardrails",
        after_guardrails,
        {"continue": "nlu", "block": "memory_update"},
    )
    g.add_conditional_edges(
        "nlu",
        after_nlu,
        {
            "retrieve": "retrieve",
            "plan": "think",
            "shortcut": "memory_update",
        },
    )
    g.add_edge("retrieve", "observe")
    g.add_edge("observe", "think")
    g.add_edge("think", "plan")
    g.add_edge("plan", "act")
    g.add_conditional_edges(
        "act",
        after_act,
        {"confirm": "confirm", "reflect": "reflect"},
    )
    g.add_edge("confirm", "memory_update")
    # replan 回到 think → plan，让用户可见第二轮「思考 → 计划 → 行动」链路
    g.add_conditional_edges(
        "reflect",
        after_reflect,
        {"replan": "think", "finish": "memory_update"},
    )
    g.add_edge("memory_update", END)

    if with_checkpointer:
        return g.compile(checkpointer=get_checkpointer())
    return g.compile()
