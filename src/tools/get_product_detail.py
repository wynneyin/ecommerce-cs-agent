"""Single-product detail tool."""

from __future__ import annotations

from typing import Any

from src.tools.data_loader import index_products


def get_product_detail(product_id: str | None = None) -> dict[str, Any]:
    if not product_id:
        return {"ok": False, "error": "missing product_id"}
    p = index_products().get(product_id.upper())
    if not p:
        return {"ok": False, "error": f"product {product_id} not found"}
    return {"ok": True, "item": p}
