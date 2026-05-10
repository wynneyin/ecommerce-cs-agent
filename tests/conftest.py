"""Force deterministic offline pipeline before importing ``src`` (pytest loads this first)."""

from __future__ import annotations

import os

# Override repo .env so tests never hit remote LLM APIs.
os.environ["LLM_PROVIDER"] = "fake"
# 避免本地 .env 中 TAVILY_API_KEY 导致 web_search 走真实 Tavily（测试只 mock DuckDuckGo）。
os.environ["WEB_SEARCH_BACKEND"] = "duckduckgo"
