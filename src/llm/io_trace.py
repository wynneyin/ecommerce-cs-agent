"""Optional stderr logging + state patches so LLM outputs are visible when debugging.

* Set ``AGENT_LOG_LLM=0`` to silence stderr (terminal / uvicorn log).
* Final state field ``llm_run_summary`` lists each remote call with response excerpt
  for Streamlit / API consumers.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from src.state import AgentState


def stderr_llm_enabled() -> bool:
    raw = os.getenv("AGENT_LOG_LLM", "1").strip().lower()
    return raw in ("1", "true", "yes", "on")


def emit_llm_stderr(stage: str, response: str, *, prompt_hint: str = "") -> None:
    """Print-like visibility: writes to stderr so uvicorn/streamlit capture shows it."""
    if not stderr_llm_enabled():
        return
    hint = (prompt_hint or "").strip()[:2000]
    body = (response or "").strip()[:16000]
    sys.stderr.write(
        f"\n{'=' * 56}\n[LLM {stage}]\n"
        f"-- prompt (hint) --\n{hint}\n"
        f"-- model response --\n{body}\n"
        f"{'=' * 56}\n"
    )


def merge_llm_summary(
    state: AgentState,
    stage: str,
    response: str,
    *,
    prompt_hint: str = "",
) -> dict[str, Any]:
    """Append one exchange; merge into node return dict."""
    emit_llm_stderr(stage, response, prompt_hint=prompt_hint)
    prev = list(state.get("llm_run_summary") or [])
    prev.append(
        {
            "stage": stage,
            "prompt_hint": (prompt_hint or "")[:800],
            "response": (response or "")[:5000],
        }
    )
    return {"llm_run_summary": prev}
