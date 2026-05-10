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


SETTINGS = Settings()
