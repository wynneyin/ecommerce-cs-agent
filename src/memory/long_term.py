"""Long-term per-user memory backed by SQLite (key/value JSON blob)."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from typing import Any

from src.config import SETTINGS


_DDL = """
CREATE TABLE IF NOT EXISTS long_memory (
    user_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class LongTermMemory:
    """Thread-safe SQLite-backed user-preference store."""

    def __init__(self, path: str | None = None):
        self.path = path or SETTINGS.long_memory_db
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_DDL)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, check_same_thread=False)

    def get(self, user_id: str) -> dict[str, Any]:
        if not user_id:
            return {}
        with self._lock, self._connect() as conn:
            cur = conn.execute("SELECT payload FROM long_memory WHERE user_id=?", (user_id,))
            row = cur.fetchone()
        if not row:
            return {}
        try:
            return json.loads(row[0])
        except Exception:
            return {}

    def put(self, user_id: str, payload: dict[str, Any]) -> None:
        if not user_id:
            return
        data = json.dumps(payload, ensure_ascii=False)
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO long_memory(user_id, payload) VALUES(?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET payload=excluded.payload, updated_at=datetime('now')",
                (user_id, data),
            )
            conn.commit()

    def clear(self, user_id: str | None = None) -> None:
        with self._lock, self._connect() as conn:
            if user_id:
                conn.execute("DELETE FROM long_memory WHERE user_id=?", (user_id,))
            else:
                conn.execute("DELETE FROM long_memory")
            conn.commit()
