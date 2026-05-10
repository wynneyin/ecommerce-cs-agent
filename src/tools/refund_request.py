"""Refund request tool (sensitive — requires HITL confirmation)."""

from __future__ import annotations

from typing import Any

from src.tools.data_loader import index_orders


SENSITIVE = True  # consumed by the confirm node


def refund_request(
    order_id: str | None = None,
    reason: str | None = None,
    confirmed: bool = False,
    user_id: str | None = None,
) -> dict[str, Any]:
    if not order_id:
        return {"ok": False, "error": "missing order_id"}
    order = index_orders().get(order_id)
    if not order:
        return {"ok": False, "error": f"order {order_id} not found"}
    if user_id and order.get("user_id") and order["user_id"] != user_id:
        return {"ok": False, "error": "order does not belong to this user"}
    if order["status"] in {"pending", "cancelled"}:
        return {"ok": False, "error": f"order status `{order['status']}` cannot be refunded"}

    if not confirmed:
        # The node layer will route to confirm before re-invoking with confirmed=True.
        return {
            "ok": False,
            "needs_confirmation": True,
            "preview": {
                "order_id": order_id,
                "amount": order["amount"],
                "product_name": order["product_name"],
                "reason": reason or "未填写",
                "warning": "退款将原路返回,通常 1-7 个工作日到账。是否确认提交?",
            },
        }

    return {
        "ok": True,
        "refund_id": f"R{order_id[1:]}",
        "order_id": order_id,
        "amount": order["amount"],
        "reason": reason or "未填写",
        "estimated_arrival_days": "1-7",
        "message": (
            f"已为订单 {order_id} 提交退款申请,金额 {order['amount']} 元,"
            f"理由:{reason or '未填写'}。预计 1-7 个工作日内退款到账。"
        ),
    }
