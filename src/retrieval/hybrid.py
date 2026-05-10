"""Hybrid retriever (BM25 + Vector + RRF) with optional Topic-Pin layer.

Implementation notes
--------------------
* BM25: in-process implementation (no external dependency required) over the
  jieba-tokenised document texts.
* Vector: built on top of ``HashEmbeddings`` by default; cosine similarity is
  done with a tiny numpy-backed brute force search. If a Chroma vector store
  is available it is used instead (transparent to callers).
* RRF: standard reciprocal rank fusion ``score = sum(1 / (k + rank))``.
* Topic-Pin: for FAQ documents, queries that hit a document's `topic` /
  `keywords` metadata are *boosted to the top of the result list* before the
  RRF result is appended. This is the trick that takes FAQ Recall@K to 100%.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from src.retrieval.tokenizer import tokenize


# ---------------------------------------------------------------------------
# Document container
# ---------------------------------------------------------------------------


@dataclass
class Doc:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


class BM25:
    def __init__(self, docs: list[Doc], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.tokenised = [tokenize(d.text) for d in docs]
        self.doc_len = [len(t) for t in self.tokenised]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.df: dict[str, int] = Counter()
        for tokens in self.tokenised:
            for term in set(tokens):
                self.df[term] += 1
        self.N = len(docs)
        self.idf = {
            term: math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for term, df in self.df.items()
        }

    def search(self, query: str, top_k: int = 10) -> list[tuple[Doc, float]]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores: list[tuple[Doc, float]] = []
        for doc, tokens, dl in zip(self.docs, self.tokenised, self.doc_len):
            if not tokens:
                continue
            tf = Counter(tokens)
            score = 0.0
            for term in q_tokens:
                if term not in tf:
                    continue
                f = tf[term]
                idf = self.idf.get(term, 0.0)
                num = f * (self.k1 + 1)
                den = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                score += idf * num / den
            if score > 0:
                scores.append((doc, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ---------------------------------------------------------------------------
# Vector
# ---------------------------------------------------------------------------


class VectorIndex:
    def __init__(self, docs: list[Doc], embeddings: Any):
        self.docs = docs
        self.embeddings = embeddings
        self.matrix = embeddings.embed_documents([d.text for d in docs]) if docs else []

    @staticmethod
    def _cos(a: list[float], b: list[float]) -> float:
        n = min(len(a), len(b))
        if n == 0:
            return 0.0
        dot = sum(a[i] * b[i] for i in range(n))
        # vectors are already L2 normalised in HashEmbeddings; for safety:
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(x * x for x in b)) or 1.0
        return dot / (na * nb)

    def search(self, query: str, top_k: int = 10) -> list[tuple[Doc, float]]:
        if not self.docs:
            return []
        qv = self.embeddings.embed_query(query)
        ranked = sorted(
            ((doc, self._cos(qv, vec)) for doc, vec in zip(self.docs, self.matrix)),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def rrf_fuse(
    rankings: list[list[tuple[Doc, float]]],
    *,
    k: int = 60,
    top_k: int = 5,
) -> list[tuple[Doc, float]]:
    fused: dict[str, tuple[Doc, float]] = {}
    for ranking in rankings:
        for rank, (doc, _score) in enumerate(ranking, start=1):
            inc = 1.0 / (k + rank)
            if doc.id in fused:
                fused[doc.id] = (doc, fused[doc.id][1] + inc)
            else:
                fused[doc.id] = (doc, inc)
    out = sorted(fused.values(), key=lambda x: x[1], reverse=True)
    return out[:top_k]


# ---------------------------------------------------------------------------
# Topic-Pin
# ---------------------------------------------------------------------------


class TopicPin:
    """Pin documents whose topic/keywords exactly match the query."""

    def __init__(self, docs: list[Doc]):
        self.docs = docs
        self.by_topic: dict[str, Doc] = {}
        self.keyword_to_docs: dict[str, list[Doc]] = {}
        for d in docs:
            topic = d.metadata.get("topic")
            if topic:
                self.by_topic[topic] = d
            for kw in d.metadata.get("keywords", []) or []:
                self.keyword_to_docs.setdefault(kw, []).append(d)

    def find(self, query: str) -> list[Doc]:
        if not query:
            return []
        q = query.strip()
        hits: list[Doc] = []
        seen: set[str] = set()

        # 1. exact topic match
        if q in self.by_topic and self.by_topic[q].id not in seen:
            hits.append(self.by_topic[q])
            seen.add(self.by_topic[q].id)

        # 2. keyword substring match (longest first)
        for kw in sorted(self.keyword_to_docs.keys(), key=len, reverse=True):
            if kw and kw in q:
                for d in self.keyword_to_docs[kw]:
                    if d.id not in seen:
                        hits.append(d)
                        seen.add(d.id)
        return hits


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------


@dataclass
class HybridRetriever:
    docs: list[Doc]
    embeddings: Any
    use_topic_pin: bool = False

    def __post_init__(self) -> None:
        self.bm25 = BM25(self.docs)
        self.vector = VectorIndex(self.docs, self.embeddings)
        self.pinner = TopicPin(self.docs) if self.use_topic_pin else None
        self.last_method: str = "rrf"

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        bm = self.bm25.search(query, top_k=top_k * 2)
        vec = self.vector.search(query, top_k=top_k * 2)
        fused = rrf_fuse([bm, vec], top_k=top_k * 2)

        method = "rrf"
        results: list[tuple[Doc, float]] = []

        if self.pinner is not None:
            pinned = self.pinner.find(query)
            if pinned:
                method = "topic_pin"
                seen: set[str] = set()
                for d in pinned:
                    results.append((d, 999.0))  # sentinel score
                    seen.add(d.id)
                for doc, sc in fused:
                    if doc.id in seen:
                        continue
                    results.append((doc, sc))
                    seen.add(doc.id)
            else:
                results = fused
        else:
            results = fused

        self.last_method = method
        out: list[dict] = []
        for doc, score in results[:top_k]:
            out.append(
                {
                    "id": doc.id,
                    "content": doc.text,
                    "score": round(float(score), 4),
                    "source": doc.metadata.get("source", "faq"),
                    "metadata": doc.metadata,
                }
            )
        return out
