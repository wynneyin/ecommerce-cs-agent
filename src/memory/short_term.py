"""Short-term memory wrapper around LangGraph checkpointers.

Returns a checkpoint saver that is safe to plug into ``graph.compile``.
On environments where the SQLite saver is not available we fall back to
``MemorySaver`` (in-process).
"""

from __future__ import annotations

import os
from typing import Any

from src.config import SETTINGS


def get_checkpointer() -> Any:
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore

        os.makedirs(os.path.dirname(SETTINGS.checkpoint_db) or ".", exist_ok=True)
        # Newer API: from_conn_string returns a context manager — we prefer
        # the direct constructor with a sqlite3 connection for stability.
        import sqlite3

        conn = sqlite3.connect(SETTINGS.checkpoint_db, check_same_thread=False)
        return SqliteSaver(conn)
    except Exception:
        from langgraph.checkpoint.memory import MemorySaver  # type: ignore

        return MemorySaver()
