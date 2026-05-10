"""Order query tool."""

from __future__ import annotations

from typing import Any

from src.tools.data_loader import index_orders


STATUS_LABEL = {
    "pending": "待付款",
    "paid": "已付款,等待发货",
    "shipped": "已发货,运输中",
    "delivered": "已签收",
    "after_sale": "售后处理中",
    "cancelled": "已取消",
}


def query_order(order_id: str | None = None, user_id: str | None = None) -> dict[str, Any]:
    if not order_id:
        return {"ok": False, "error": "missing order_id"}
    order = index_orders().get(order_id)
    if not order:
        return {"ok": False, "error": f"order {order_id} not found"}
    if user_id and order.get("user_id") and order["user_id"] != user_id:
        return {"ok": False, "error": "order does not belong to this user"}

    enriched = dict(order)
    enriched["status_label"] = STATUS_LABEL.get(order["status"], order["status"])
    return {"ok": True, "order": enriched}
