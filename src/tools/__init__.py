"""Tool registry consumed by the `act` node."""

from __future__ import annotations

from typing import Any, Callable

from src.tools.compare_products import compare_products
from src.tools.faq_retrieve import faq_retrieve
from src.tools.get_product_detail import get_product_detail
from src.tools.query_order import query_order
from src.tools.refund_request import refund_request
from src.tools.search_products import search_products


TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "search_products": search_products,
    "get_product_detail": get_product_detail,
    "compare_products": compare_products,
    "query_order": query_order,
    "refund_request": refund_request,
    "faq_retrieve": faq_retrieve,
}

SENSITIVE_TOOLS = {"refund_request"}


def get_tool(name: str) -> Callable[..., dict[str, Any]] | None:
    return TOOL_REGISTRY.get(name)


__all__ = ["TOOL_REGISTRY", "SENSITIVE_TOOLS", "get_tool"]
