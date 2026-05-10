"""Minimal FastAPI wrapper around the LangGraph agent.

Run::

    uvicorn apps.api_server:app --reload --port 8000

Endpoints
---------
* ``GET  /health`` — liveness probe
* ``POST /chat`` — execute one turn

  Request body::

    {
        "user_input": "推荐 5000 元的笔记本",
        "user_id": "demo",
        "thread_id": "abc",
        "mode": "deterministic",
        "use_memory": true,
        "confirm_decision": null
    }

  Response::

    {
        "final_response": "...",
        "intent": "...",
        "slots": {...},
        "actions": [...],
        "confirm_required": false,
        "trace": [...]
    }
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph import run_turn  # noqa: E402

app = FastAPI(title="Ecommerce CS Agent")


class ChatRequest(BaseModel):
    user_input: str
    user_id: str = "anon"
    thread_id: Optional[str] = None
    mode: str = "deterministic"
    use_memory: bool = True
    confirm_decision: Optional[str] = None


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    state = run_turn(
        req.user_input,
        user_id=req.user_id,
        thread_id=req.thread_id,
        mode=req.mode,
        use_memory=req.use_memory,
        confirm_decision=req.confirm_decision,
    )
    return {
        "final_response": state.get("final_response"),
        "intent": state.get("intent"),
        "intent_conf": state.get("intent_conf"),
        "slots": state.get("slots"),
        "retrieval_method": state.get("retrieval_method"),
        "actions": [
            {
                "name": a.get("name"),
                "args": a.get("args"),
                "ok": a.get("ok"),
                "latency_ms": a.get("latency_ms"),
            }
            for a in (state.get("actions") or [])
        ],
        "confirm_required": state.get("confirm_required"),
        "confirm_payload": state.get("confirm_payload"),
        "memory_long": state.get("memory_long"),
        "trace": state.get("trace"),
    }
