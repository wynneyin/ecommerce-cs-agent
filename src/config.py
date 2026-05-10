"""Central runtime configuration.

All knobs are read from environment variables (with sensible defaults) so the
project can run end-to-end with zero external services (`LLM_PROVIDER=fake`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # python-dotenv is optional
    pass


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
CACHE_DIR = PROJECT_ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _bool(name: str, default: bool = False) -> bool:
    raw = _env(name, "1" if default else "0").lower()
    return raw in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


@dataclass
class Settings:
    # LLM
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "fake"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", ""))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", ""))
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", ""))

    # Embeddings
    embedding_provider: str = field(default_factory=lambda: _env("EMBEDDING_PROVIDER", "fake"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", ""))

    # 联网搜索：设 TAVILY_API_KEY 则默认走 Tavily（见 https://www.tavily.com/ ）；WEB_SEARCH_BACKEND=duckduckgo 可强制 DuckDuckGo
    tavily_api_key: str = field(default_factory=lambda: _env("TAVILY_API_KEY", ""))

    # Tracing
    langsmith_enabled: bool = field(default_factory=lambda: _bool("LANGSMITH_TRACING", False))
    langsmith_project: str = field(
        default_factory=lambda: _env("LANGSMITH_PROJECT", "ecommerce-cs-agent")
    )

    # Persistence paths
    checkpoint_db: str = field(
        default_factory=lambda: _env("CHECKPOINT_DB", str(CACHE_DIR / "checkpoints.sqlite"))
    )
    long_memory_db: str = field(
        default_factory=lambda: _env("LONG_MEMORY_DB", str(CACHE_DIR / "long_memory.sqlite"))
    )
    chroma_dir: str = field(
        default_factory=lambda: _env("CHROMA_DIR", str(CACHE_DIR / "chroma"))
    )

    # Defaults
    default_top_k: int = 5
    default_mode: str = "deterministic"

    # Hybrid retrieval: dual recall score fusion (env overrides; 和归一化为 1)
    retrieval_vec_weight: float = field(default_factory=lambda: _float("RETRIEVAL_VEC_WEIGHT", 0.7))
    retrieval_bm25_weight: float = field(default_factory=lambda: _float("RETRIEVAL_BM25_WEIGHT", 0.3))
    # 非空则对加权融合后的前若干条用 CrossEncoder 重排（需 sentence-transformers，首次会下载模型）
    reranker_model: str = field(default_factory=lambda: _env("RERANKER_MODEL", ""))

    def is_fake_llm(self) -> bool:
        return self.llm_provider.lower() == "fake"

    def is_fake_embedding(self) -> bool:
        return self.embedding_provider.lower() == "fake"

    def use_llm_nlu(self) -> bool:
        """Use remote NLU when not on fake provider (overridable via USE_LLM_NLU)."""
        if self.is_fake_llm():
            return False
        return _bool("USE_LLM_NLU", True)

    def use_llm_thinking(self) -> bool:
        """Reasoning step between observe and plan (USE_LLM_THINKING)."""
        if self.is_fake_llm():
            return False
        return _bool("USE_LLM_THINKING", True)

    def use_llm_synthesis(self) -> bool:
        """Compose final reply with the chat model (USE_LLM_SYNTHESIS)."""
        if self.is_fake_llm():
            return False
        return _bool("USE_LLM_SYNTHESIS", True)

    def use_llm_react_reflect(self) -> bool:
        """After act, generate readable ReAct-style reflection text (USE_LLM_REACT_REFLECT)."""
        if self.is_fake_llm():
            return False
        return _bool("USE_LLM_REACT_REFLECT", True)

    def use_llm_conversational_fallback(self) -> bool:
        """When no tool JSON reply, paraphrase template draft into natural CS tone (USE_LLM_CONVERSATIONAL)."""
        if self.is_fake_llm():
            return False
        return _bool("USE_LLM_CONVERSATIONAL", True)

    def use_llm_query_understanding(self) -> bool:
        """Guardrails → NLU 之前：用大模型拆解用户问题（USE_LLM_UNDERSTAND）。"""
        if self.is_fake_llm():
            return False
        return _bool("USE_LLM_UNDERSTAND", True)

    def use_llm_web_tool(self) -> bool:
        """Plan 阶段由模型决定是否前置 ``web_search``（USE_LLM_WEB_TOOL）；fake 关闭。"""
        if self.is_fake_llm():
            return False
        return _bool("USE_LLM_WEB_TOOL", True)

    def web_search_backend(self) -> str:
        """``tavily`` 或 ``duckduckgo``。有 Tavily 密钥且未强制 duckduckgo 时用 Tavily。"""
        b = _env("WEB_SEARCH_BACKEND", "").lower()
        if b == "duckduckgo":
            return "duckduckgo"
        if b == "tavily":
            return "tavily" if self.tavily_api_key else "duckduckgo"
        return "tavily" if self.tavily_api_key else "duckduckgo"


SETTINGS = Settings()
