"""Internet search — Tavily（推荐，需 API Key）或 DuckDuckGo 兜底。

Tavily 面向 LLM/Agent，结构化结果与摘要便于 RAG；注册与用量见官方文档。
社区介绍可参考： https://agent.csdn.net/683fae0a606a8318e85aece0.html

* ``pip install tavily-python``，环境变量 ``TAVILY_API_KEY=tvly-...``
* 未配置密钥时回退 ``duckduckgo-search``（无需 Key）
"""

from __future__ import annotations

from typing import Any

from src.config import SETTINGS

try:
    from duckduckgo_search import DDGS as _DDGS  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - optional install
    _DDGS = None

try:
    from tavily import TavilyClient  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    TavilyClient = None  # type: ignore[misc, assignment]


def _normalize_items_to_summary(items: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, it in enumerate(items, start=1):
        title = it.get("title") or "(无标题)"
        snip = it.get("snippet") or ""
        lines.append(f"{i}. {title}\n   {snip[:280]}{'…' if len(snip) > 280 else ''}")
    return "\n".join(lines)


def _search_tavily(q: str, cap: int) -> dict[str, Any]:
    if TavilyClient is None:
        return {
            "ok": False,
            "error": "missing_dependency",
            "hint": "pip install tavily-python",
            "items": [],
            "query": q,
        }
    key = SETTINGS.tavily_api_key
    if not key:
        return {"ok": False, "error": "missing_tavily_api_key", "items": [], "query": q}

    items: list[dict[str, Any]] = []
    try:
        client = TavilyClient(api_key=key)
        resp = client.search(
            q,
            max_results=cap,
            include_answer=True,
            search_depth="basic",
        )
    except Exception as exc:  # pragma: no cover - network
        return {"ok": False, "error": repr(exc), "items": [], "query": q}

    answer = (resp.get("answer") or "").strip()
    for r in resp.get("results") or []:
        if not isinstance(r, dict):
            continue
        items.append(
            {
                "title": (r.get("title") or "").strip(),
                "url": (r.get("url") or "").strip(),
                "snippet": (r.get("content") or r.get("raw_content") or "").strip(),
            }
        )

    if not items and not answer:
        return {
            "ok": True,
            "query": q,
            "items": [],
            "summary": "未检索到匹配结果，请换个关键词试试。",
            "provider": "tavily",
        }

    summary = _normalize_items_to_summary(items)
    if answer:
        summary = f"【摘要】{answer}\n\n{summary}" if summary else f"【摘要】{answer}"

    return {
        "ok": True,
        "query": q,
        "items": items,
        "summary": summary,
        "provider": "tavily",
    }


def _search_duckduckgo(q: str, cap: int) -> dict[str, Any]:
    if _DDGS is None:
        return {
            "ok": False,
            "error": "missing_dependency",
            "hint": "pip install duckduckgo-search",
            "items": [],
            "query": q,
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
    except Exception as exc:  # pragma: no cover
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
            "provider": "duckduckgo",
        }

    return {
        "ok": True,
        "query": q,
        "items": items,
        "summary": _normalize_items_to_summary(items),
        "provider": "duckduckgo",
    }


def web_search(*, query: str, max_results: int = 5) -> dict[str, Any]:
    """联网检索：优先 Tavily（配置 ``TAVILY_API_KEY``），否则 DuckDuckGo。"""
    q = (query or "").strip()
    if not q:
        return {"ok": False, "error": "empty_query", "items": []}

    cap = max(1, min(int(max_results), 10))
    backend = SETTINGS.web_search_backend()

    if backend == "tavily":
        out = _search_tavily(q, cap)
        if out.get("ok"):
            return out
        # 密钥缺失、依赖未装、网络/额度等：统一回退 DuckDuckGo，尽量给出可检索结果
        return _search_duckduckgo(q, cap)

    return _search_duckduckgo(q, cap)
