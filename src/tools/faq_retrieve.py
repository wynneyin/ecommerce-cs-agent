"""FAQ retrieval tool — thin wrapper over the hybrid retriever.

The retriever module is imported lazily to avoid a circular import:
``src.tools.__init__`` -> ``faq_retrieve`` -> ``src.retrieval`` -> ``data_loader``.
"""

from __future__ import annotations

from typing import Any


def faq_retrieve(query: str | None = None, top_k: int = 5) -> dict[str, Any]:
    if not query:
        return {"ok": False, "error": "missing query", "items": []}
    from src.retrieval import get_retriever  # local import (anti-circular)

    retriever = get_retriever("faq")
    docs = retriever.retrieve(query, top_k=top_k)
    return {
        "ok": True,
        "items": docs,
        "method": retriever.last_method,
    }
