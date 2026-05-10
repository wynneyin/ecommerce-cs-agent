"""Product search tool."""

from __future__ import annotations

from typing import Any

from src.tools.data_loader import load_products


def search_products(
    query: str | None = None,
    category: str | None = None,
    budget: int | float | None = None,
    tags: list[str] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Search the catalogue.

    Returns a dict with `items` (list of product summaries) and `total` count.
    Scoring is a simple weighted sum of:
    * exact-token query overlap with name/tags/specs/category
    * tag matches
    * budget fit (price <= budget gets a bonus, big mismatches get a penalty)
    """
    items: list[tuple[float, dict]] = []
    q = (query or "").lower()
    qtokens = [t for t in q.replace(",", " ").split() if t]

    for p in load_products():
        if category and p["category"] != category:
            continue
        score = 0.0
        haystack = (
            p["name"].lower()
            + " "
            + " ".join(map(str, p.get("tags", []))).lower()
            + " "
            + " ".join(map(str, p.get("specs", []))).lower()
        )
        for t in qtokens:
            if t and t in haystack:
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

        if score > 0 or category or tags:
            items.append((score, p))

    items.sort(key=lambda x: x[0], reverse=True)
    top = items[:top_k]
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
    }
