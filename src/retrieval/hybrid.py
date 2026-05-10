"""Hybrid retriever：BM25 + 向量双路召回，加权融合 + 可选 CrossEncoder 重排 + Topic-Pin。

* **BM25**：jieba 分词后的 in-process 检索。
* **向量**：默认 ``HashEmbeddings`` + 余弦；可换真实 embedding 做语义召回。
* **融合**：各路分数 min-max 归一化后线性加权，默认 **向量 0.7 + BM25 0.3**（``RETRIEVAL_VEC_WEIGHT`` / ``RETRIEVAL_BM25_WEIGHT``）。
* **RRF**：仍保留函数 ``rrf_fuse`` 供对比或外部调用。
* **Rerank**：若设置 ``RERANKER_MODEL``（如 cross-encoder 模型名），对融合后前若干条做 CrossEncoder 打分重排。
* **Topic-Pin**（FAQ）：命中 topic/keywords 的文档置顶，其后接融合（及 rerank）结果。
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from src.config import SETTINGS
from src.retrieval.tokenizer import tokenize

_cross_encoder_cache: dict[str, Any] = {}


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
# Weighted score fusion (vector + BM25)
# ---------------------------------------------------------------------------


def _min_max_norm(score_by_id: dict[str, float]) -> dict[str, float]:
    if not score_by_id:
        return {}
    vals = list(score_by_id.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {k: 1.0 for k in score_by_id}
    return {k: (v - lo) / (hi - lo) for k, v in score_by_id.items()}


def weighted_dual_fuse(
    bm25_ranked: list[tuple[Doc, float]],
    vec_ranked: list[tuple[Doc, float]],
    *,
    vec_weight: float,
    bm25_weight: float,
    pool_limit: int,
) -> list[tuple[Doc, float]]:
    """双路分数各自 min-max 后加权求和；仅出现在一路上的文档，另一路视为 0。"""
    bm_map = {doc.id: sc for doc, sc in bm25_ranked}
    vec_map = {doc.id: sc for doc, sc in vec_ranked}
    id_to_doc: dict[str, Doc] = {}
    for doc, _ in bm25_ranked:
        id_to_doc[doc.id] = doc
    for doc, _ in vec_ranked:
        id_to_doc[doc.id] = doc

    bm_n = _min_max_norm(dict(bm_map))
    vec_n = _min_max_norm(dict(vec_map))
    all_ids = set(bm_map) | set(vec_map)
    fused: list[tuple[Doc, float]] = []
    for did in all_ids:
        doc = id_to_doc[did]
        v = vec_n.get(did, 0.0)
        b = bm_n.get(did, 0.0)
        fused.append((doc, vec_weight * v + bm25_weight * b))
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:pool_limit]


def _get_cross_encoder(model_name: str) -> Any:
    if model_name not in _cross_encoder_cache:
        from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]

        _cross_encoder_cache[model_name] = CrossEncoder(model_name)
    return _cross_encoder_cache[model_name]


def cross_encoder_rerank(
    query: str,
    ranked: list[tuple[Doc, float]],
    *,
    model_name: str,
    top_k: int,
) -> list[tuple[Doc, float]]:
    """对前若干条用 CrossEncoder(query, doc) 重排；分数为归一化后的相关性分。"""
    if not model_name or len(ranked) <= 1:
        return ranked
    pool_n = min(len(ranked), max(top_k * 4, 20))
    pool = ranked[:pool_n]
    tail = ranked[pool_n:]
    try:
        ce = _get_cross_encoder(model_name)
    except Exception:
        return ranked
    pairs = [[query, d.text[:2000]] for d, _ in pool]
    try:
        raw = ce.predict(pairs, show_progress_bar=False, batch_size=16)
    except Exception:
        return ranked
    raw_list = [float(x) for x in raw]
    rn = _min_max_norm({pool[i][0].id: raw_list[i] for i in range(len(pool))})
    order = sorted(range(len(pool)), key=lambda i: raw_list[i], reverse=True)
    new_pool = [(pool[i][0], rn[pool[i][0].id]) for i in order]
    return new_pool + tail


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
        self.last_method: str = "weighted_fusion"

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        recall_k = max(top_k * 2, 10)
        bm = self.bm25.search(query, top_k=recall_k)
        vec = self.vector.search(query, top_k=recall_k)

        w_vec = float(SETTINGS.retrieval_vec_weight)
        w_bm = float(SETTINGS.retrieval_bm25_weight)
        s = w_vec + w_bm
        if s > 1e-9:
            w_vec, w_bm = w_vec / s, w_bm / s

        pool_limit = max(recall_k, top_k * 2)
        fused = weighted_dual_fuse(
            bm,
            vec,
            vec_weight=w_vec,
            bm25_weight=w_bm,
            pool_limit=pool_limit,
        )

        method = "weighted_fusion"
        rerank_model = (SETTINGS.reranker_model or "").strip()
        if rerank_model:
            fused = cross_encoder_rerank(
                query, fused, model_name=rerank_model, top_k=top_k
            )
            method = "weighted_fusion+cross_encoder_rerank"

        results: list[tuple[Doc, float]] = []

        if self.pinner is not None:
            pinned = self.pinner.find(query)
            if pinned:
                method = f"topic_pin+{method}"
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
