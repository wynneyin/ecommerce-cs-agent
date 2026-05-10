"""Graph factory + a high-level ``run_turn`` helper."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from src.graph.deterministic_graph import build_deterministic_graph
from src.graph.react_graph import build_react_graph
from src.memory import LongTermMemory
from src.state import AgentState, initial_state


@lru_cache(maxsize=4)
def get_graph(mode: str = "deterministic", with_checkpointer: bool = True):
    if mode == "react":
        return build_react_graph(with_checkpointer=with_checkpointer)
    return build_deterministic_graph(with_checkpointer=with_checkpointer)


_LONG_MEM = LongTermMemory()


def run_turn(
    user_input: str,
    *,
    user_id: str = "anon",
    thread_id: str | None = None,
    mode: str = "deterministic",
    use_memory: bool = True,
    confirm_decision: str | None = None,
    extra_state: dict[str, Any] | None = None,
) -> AgentState:
    """Run a single turn and return the final state.

    ``thread_id`` is forwarded to the checkpointer; reusing the same value
    enables short-term memory across turns. When ``use_memory`` is False the
    long-term memory is not loaded (used to compute the resume's *Memory Off*
    baseline).
    """
    graph = get_graph(mode, with_checkpointer=True)
    state = initial_state(user_input, user_id=user_id, mode=mode)

    if use_memory:
        state["memory_long"] = _LONG_MEM.get(user_id)
    else:
        state["memory_long"] = {}

    if confirm_decision is not None:
        state["confirm_decision"] = confirm_decision  # type: ignore[assignment]

    if extra_state:
        state.update(extra_state)  # type: ignore[arg-type]

    config: dict[str, Any] = {}
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}
    else:
        config["configurable"] = {"thread_id": state["run_id"]}

    final_state = graph.invoke(state, config=config)
    return final_state  # type: ignore[return-value]


__all__ = [
    "build_deterministic_graph",
    "build_react_graph",
    "get_graph",
    "run_turn",
]
