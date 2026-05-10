"""Factory for LLM and embedding objects.

Defaults to a fully offline ``FakeChatModel`` / ``FakeEmbeddings`` so the
project runs out of the box. Real providers are imported lazily so the package
has no hard dependency on them.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Any

from src.config import SETTINGS


# ---------------------------------------------------------------------------
# Chat model
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_chat_model(provider: str | None = None) -> Any:
    provider = (provider or SETTINGS.llm_provider or "fake").lower()
    if provider == "fake":
        from src.llm.fake_llm import FakeChatModel

        return FakeChatModel()
    if provider == "openai":
        from langchain_openai import ChatOpenAI  # type: ignore

        return ChatOpenAI(
            model=SETTINGS.llm_model or "gpt-4o-mini",
            api_key=SETTINGS.llm_api_key or None,
            base_url=SETTINGS.llm_base_url or None,
            temperature=0.2,
        )
    if provider == "deepseek":
        from langchain_openai import ChatOpenAI  # type: ignore

        return ChatOpenAI(
            model=SETTINGS.llm_model or "deepseek-chat",
            api_key=SETTINGS.llm_api_key or None,
            base_url=SETTINGS.llm_base_url or "https://api.deepseek.com/v1",
            temperature=0.2,
        )
    if provider == "ollama":
        from langchain_community.chat_models import ChatOllama  # type: ignore

        return ChatOllama(model=SETTINGS.llm_model or "qwen2.5", temperature=0.2)
    raise ValueError(f"Unknown LLM provider: {provider}")


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class HashEmbeddings:
    """Cheap deterministic embedding for offline tests.

    Uses MD5 hashes of (token, position) bigrams projected into a fixed-size
    vector. Not great for semantic search but consistent and free.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = list(text)
        for i, ch in enumerate(tokens):
            h = hashlib.md5(f"{ch}-{i % 7}".encode("utf-8")).digest()
            for j in range(0, len(h), 2):
                idx = (h[j] * 256 + h[j + 1]) % self.dim
                vec[idx] += 1.0
        # L2 normalise
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@lru_cache(maxsize=1)
def get_embeddings(provider: str | None = None) -> Any:
    provider = (provider or SETTINGS.embedding_provider or "fake").lower()
    if provider == "fake":
        return HashEmbeddings()
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings  # type: ignore

        return OpenAIEmbeddings(
            model=SETTINGS.embedding_model or "text-embedding-3-small",
            api_key=SETTINGS.llm_api_key or None,
        )
    if provider == "bge":
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore

        return HuggingFaceEmbeddings(model_name=SETTINGS.embedding_model or "BAAI/bge-m3")
    if provider == "ollama":
        from langchain_community.embeddings import OllamaEmbeddings  # type: ignore

        return OllamaEmbeddings(model=SETTINGS.embedding_model or "bge-m3")
    raise ValueError(f"Unknown embedding provider: {provider}")
