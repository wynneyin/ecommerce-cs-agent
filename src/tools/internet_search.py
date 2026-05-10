"""Internet search tool — DuckDuckGo text results (no API key).

Requires optional dependency ``duckduckgo-search``. Install::

    pip install duckduckgo-search
"""

from __future__ import annotations

from typing import Any

try:
    from duckduckgo_search import DDGS as _DDGS  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - optional install
    _DDGS = None


def web_search(*, query: str, max_results: int = 5) -> dict[str, Any]:
    """Return compact web snippets for downstream synthesis."""
    q = (query or "").strip()
    if not q:
        return {"ok": False, "error": "empty_query", "items": []}

    cap = max(1, min(int(max_results), 10))

    if _DDGS is None:
        return {
            "ok": False,
            "error": "missing_dependency",
            "hint": "pip install duckduckgo-search",
            "items": [],
        }

    items: list[dict[str, Any]] = []
    try:
        with _DDGS() as ddgs:
            for r in ddgs.text(q, max_results=cap):
                items.append(
                    {
                        "title": (r.get("title") or "").strip(),
                        "url": (r.get("href") or "").strip(),
                        "snippet": (r.get("body") or "").strip(),
                    }
                )
    except Exception as exc:  # pragma: no cover - network / upstream
        return {
            "ok": False,
            "error": repr(exc),
            "items": [],
            "query": q,
        }

    if not items:
        return {
            "ok": True,
            "query": q,
            "items": [],
            "summary": "未检索到匹配结果，请换个关键词试试。",
        }

    lines = []
    for i, it in enumerate(items, start=1):
        title = it.get("title") or "(无标题)"
        snip = it.get("snippet") or ""
        lines.append(f"{i}. {title}\n   {snip[:280]}{'…' if len(snip) > 280 else ''}")

    return {
        "ok": True,
        "query": q,
        "items": items,
        "summary": "\n".join(lines),
    }
