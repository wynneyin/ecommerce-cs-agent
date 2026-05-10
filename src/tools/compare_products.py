"""Compare two or more products side by side."""

from __future__ import annotations

from typing import Any

from src.tools.data_loader import index_products


def compare_products(product_ids: list[str] | None = None) -> dict[str, Any]:
    if not product_ids or len(product_ids) < 2:
        return {"ok": False, "error": "need at least 2 product_ids"}

    idx = index_products()
    items: list[dict] = []
    for pid in product_ids:
        p = idx.get(pid.upper())
        if p:
            items.append(p)

    if len(items) < 2:
        return {"ok": False, "error": "less than 2 products resolved"}

    # Build comparison table
    table = {
        "name": [p["name"] for p in items],
        "category": [p["category"] for p in items],
        "price": [p["price"] for p in items],
        "rating": [p["rating"] for p in items],
        "tags": [", ".join(p.get("tags", [])) for p in items],
        "specs": [", ".join(map(str, p.get("specs", []))) for p in items],
    }

    cheapest = min(items, key=lambda p: p["price"])
    highest_rated = max(items, key=lambda p: p["rating"])

    summary = (
        f"在所选 {len(items)} 款商品中,"
        f"价格最低的是 {cheapest['name']}({cheapest['price']} 元),"
        f"评分最高的是 {highest_rated['name']}({highest_rated['rating']} 分)。"
    )

    return {
        "ok": True,
        "items": items,
        "table": table,
        "summary": summary,
    }
