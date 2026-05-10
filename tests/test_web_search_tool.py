"""Tests for ``web_search`` tool (mocked HTTP/DuckDuckGo)."""

from __future__ import annotations

from unittest.mock import patch

from src.llm.rules import plan_for_intent, run_nlu
from src.tools.internet_search import web_search


class _FakeDDGS:
    """Minimal stand-in for duckduckgo_search.DDGS context manager."""

    def __enter__(self) -> _FakeDDGS:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def text(self, query: str, max_results: int = 5):
        return [
            {
                "title": "Example",
                "href": "https://example.com/p",
                "body": "This is a synthetic snippet for testing.",
            }
        ]


def test_run_nlu_detects_web_search_intent():
    r = run_nlu("上网搜一下 Python 发布公告")
    assert r["intent"] == "web_search"


def test_plan_maps_web_search():
    p = plan_for_intent("web_search", {"query": "天气"})
    assert p["plan"] and p["plan"][0]["name"] == "web_search"
    assert p["plan"][0]["args"]["query"] == "天气"


@patch("src.tools.internet_search._DDGS", _FakeDDGS)
def test_web_search_tool_mocked():
    out = web_search(query="unit test query", max_results=3)
    assert out.get("ok") is True
    assert out.get("items")
    assert "synthetic snippet" in (out.get("summary") or "")


@patch("src.tools.internet_search._DDGS", _FakeDDGS)
def test_run_turn_web_search_invokes_tool():
    from src.graph import run_turn

    s = run_turn(
        "联网搜索 Streamlit 组件教程",
        user_id="ws_u",
        thread_id="ws_thr",
    )
    assert s["intent"] == "web_search"
    assert s.get("actions")
    assert s["actions"][-1]["name"] == "web_search"
    assert s["actions"][-1]["ok"]
