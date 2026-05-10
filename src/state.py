"""Shared state schemas for the LangGraph agent."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# Sub-types
# ---------------------------------------------------------------------------


class ToolCall(TypedDict, total=False):
    name: str
    args: dict[str, Any]
    rationale: str


class ActionRecord(TypedDict, total=False):
    name: str
    args: dict[str, Any]
    output: Any
    ok: bool
    error: Optional[str]
    latency_ms: float


class TraceEvent(TypedDict, total=False):
    node: str
    phase: Literal["start", "end", "error"]
    timestamp: str
    payload: dict[str, Any]
    latency_ms: float


class RetrievedDoc(TypedDict, total=False):
    id: str
    content: str
    score: float
    source: str
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """The single source of truth flowing through the graph.

    Most fields are optional because the early nodes only touch a subset.
    """

    # IO
    messages: Annotated[list[BaseMessage], add_messages]
    user_input: str
    final_response: str

    # Run identifiers
    run_id: str
    turn_id: int
    user_id: str
    mode: Literal["react", "deterministic"]

    # Guardrails
    guardrails_pass: bool
    guardrails_reason: Optional[str]

    # Understand（大模型拆解用户话，早于 NLU）
    query_understanding: str

    # NLU
    intent: str
    intent_conf: float
    slots: dict[str, Any]
    nlu_timing_ms: dict[str, float]  # rules vs llm.invoke 拆分（便于排查延迟）

    # Retrieval
    retrieved: list[RetrievedDoc]
    retrieval_method: str
    retrieval_query: str

    # Plan / Act
    plan: list[ToolCall]
    actions: list[ActionRecord]
    pending_tool: Optional[ToolCall]

    # Confirmation (HITL)
    confirm_required: bool
    confirm_payload: Optional[dict[str, Any]]
    confirm_decision: Optional[Literal["approve", "reject"]]

    # Reflection
    reflection: Optional[str]
    need_replan: bool
    replan_count: int
    react_trace: list[dict[str, Any]]  # ReAct：每轮 reflect 追加一步（含可选 LLM 反思）

    # LLM reasoning (observe → plan)
    thinking: str

    # Memory
    memory_short: dict[str, Any]
    memory_working: dict[str, Any]
    memory_long: dict[str, Any]

    # Trace
    trace: list[TraceEvent]

    # LLM raw exchanges (for UI / operators; each node appends)
    llm_run_summary: list[dict[str, Any]]


# Default factory ----------------------------------------------------------


def initial_state(user_input: str, *, user_id: str = "anon", mode: str = "deterministic") -> AgentState:
    """Build a fresh AgentState for a new turn."""
    import uuid

    return AgentState(
        messages=[],
        user_input=user_input,
        final_response="",
        run_id=str(uuid.uuid4())[:8],
        turn_id=0,
        user_id=user_id,
        mode=mode,  # type: ignore[arg-type]
        guardrails_pass=True,
        guardrails_reason=None,
        query_understanding="",
        intent="unknown",
        intent_conf=0.0,
        slots={},
        retrieved=[],
        retrieval_method="none",
        retrieval_query="",
        plan=[],
        actions=[],
        pending_tool=None,
        confirm_required=False,
        confirm_payload=None,
        confirm_decision=None,
        reflection=None,
        need_replan=False,
        replan_count=0,
        react_trace=[],
        thinking="",
        memory_short={},
        memory_working={},
        memory_long={},
        trace=[],
        llm_run_summary=[],
    )
