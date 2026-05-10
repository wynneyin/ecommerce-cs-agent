"""Product search tool."""

from __future__ import annotations

from typing import Any

from src.retrieval.tokenizer import tokenize as tokenize_query
from src.tools.data_loader import index_products, load_products
from src.tools.product_taxonomy import infer_categories_from_tokens

# NLU 可能输出「笔记本电脑」等，与 catalogue 的 category 字段不一致时做归一
_CATEGORY_ALIASES: dict[str, str] = {
    "电脑": "笔记本",  # 本库无台式机；口语「买电脑」默认笔电类目
    "笔记本电脑": "笔记本",
    "轻薄本": "笔记本",
    "游戏本": "笔记本",
    "笔电": "笔记本",
    "手提电脑": "笔记本",
}


def _normalize_category(cat: str | None) -> str | None:
    if not cat:
        return None
    c = str(cat).strip()
    return _CATEGORY_ALIASES.get(c, c)


def search_products(
    query: str | None = None,
    category: str | None = None,
    budget: int | float | None = None,
    tags: list[str] | None = None,
    top_k: int = 5,
    seed_product_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Search the catalogue.

    Returns a dict with `items` (list of product summaries) and `total` count.
    Scoring is a simple weighted sum of:
    * token overlap (jieba 分词中文查询) with name/tags/specs/**category**/description
    * tag matches
    * budget fit (price <= budget gets a bonus, big mismatches get a penalty)
    * **seed_product_ids**：来自前置混合检索（BM25+向量+RRF）的 id 列表，给加分并保证入队（弥补纯关键词与语义不一致）
    """
    items: list[tuple[float, dict]] = []
    used_catalog_fallback = False  # 关键词/类目都未形成候选时，用全库评分兜底
    q = (query or "").strip()
    q_lower = q.lower()
    # 中文不能用空格 split；与检索模块一致用 jieba（或 CJK 逐字回退）
    qtokens = [t for t in tokenize_query(q) if t and len(t) >= 1]
    if not qtokens and q_lower:
        qtokens = [t for t in q_lower.replace(",", " ").split() if t]

    category = _normalize_category(category)
    # NLU 未带类目时，用语义扩展词推断（如「电脑」→ 笔记本、平板）
    inferred_cats = infer_categories_from_tokens(qtokens)
    categories_filter: set[str] | None = None
    if category:
        categories_filter = {category}
    elif inferred_cats:
        categories_filter = inferred_cats

    def _haystack(p: dict) -> str:
        parts = [
            str(p.get("name", "")),
            str(p.get("category", "")),
            " ".join(map(str, p.get("retrieval_tags", []))),
            " ".join(map(str, p.get("tags", []))),
            " ".join(map(str, p.get("specs", []))),
            str(p.get("description", "")),
        ]
        return " ".join(parts).lower()

    for p in load_products():
        if categories_filter and p["category"] not in categories_filter:
            continue
        score = 0.0
        haystack = _haystack(p)
        for t in qtokens:
            tl = t.lower()
            if tl and tl in haystack:
                score += 1.0
        if tags:
            for tag in tags:
                if tag in p.get("tags", []):
                    score += 0.7
        if budget:
            if p["price"] <= budget:
                score += 0.5
                # closer to budget = better fit
                score += (budget - p["price"]) / max(budget, 1) * 0.3
            else:
                score -= 0.5
        # quality prior
        score += (p.get("rating", 4.5) - 4.5) * 0.5

        # 有过滤或得分时保留
        if score > 0 or categories_filter or tags:
            items.append((score, p))

    # 关键词未命中且未能从查询推断类目时，按标准类目名兜底
    if not items and qtokens:
        known = {p["category"] for p in load_products() if p.get("category")}
        for t in qtokens:
            if t in known:
                categories_filter = {t}
                break
        if categories_filter:
            for p in load_products():
                if p["category"] not in categories_filter:
                    continue
                haystack = _haystack(p)
                score = sum(1.0 for x in qtokens if x.lower() in haystack)
                score += (p.get("rating", 4.5) - 4.5) * 0.5
                items.append((score, p))

    # 仍为空：返回高评分商品，避免「有库存但完全搜不到」
    if not items:
        used_catalog_fallback = True
        for p in load_products():
            if categories_filter and p["category"] not in categories_filter:
                continue
            score = (p.get("rating", 4.5) - 4.5) * 0.5
            items.append((score, p))

    # 与 retrieve 节点的向量/BM25/RRF 结果对齐：种子商品加分并入池（不过滤 category，避免 NLU 类目与语义检索打架）
    by_id: dict[str, tuple[float, dict]] = {}
    for s, p in items:
        pid = p["product_id"]
        if pid not in by_id or by_id[pid][0] < s:
            by_id[pid] = (s, p)

    idx = index_products()
    for rank, sid in enumerate((seed_product_ids or [])[:15]):
        p = idx.get(sid)
        if not p:
            continue
        boost = 12.0 - min(rank, 12) * 0.4
        old = by_id.get(sid)
        if old:
            by_id[sid] = (old[0] + boost, old[1])
        else:
            by_id[sid] = (boost, p)

    items = sorted(by_id.values(), key=lambda x: x[0], reverse=True)
    top = items[:top_k]
    max_score = max((s for s, _ in top), default=0.0)
    # strong：有明显关键词/预算等信号；soft：有结果但分偏低；browse：全库兜底或近似纯评分
    if used_catalog_fallback or not q:
        match_tier = "browse"
    elif max_score >= 1.0:
        match_tier = "strong"
    elif max_score > 0.15:
        match_tier = "soft"
    else:
        match_tier = "browse"
    suggestive = match_tier in ("soft", "browse")

    return {
        "items": [
            {
                "product_id": p["product_id"],
                "name": p["name"],
                "category": p["category"],
                "price": p["price"],
                "rating": p["rating"],
                "tags": p.get("tags", []),
                "score": round(s, 3),
            }
            for s, p in top
        ],
        "total": len(items),
        "match_tier": match_tier,
        "suggestive": suggestive,
        "used_catalog_fallback": used_catalog_fallback,
    }
