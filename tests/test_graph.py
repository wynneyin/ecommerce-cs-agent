import json
from pathlib import Path

from src.graph import run_turn

_DATA = Path(__file__).resolve().parent.parent / "data" / "orders.json"


def _first_refundable_order_id() -> str:
    orders = json.loads(_DATA.read_text(encoding="utf-8"))
    for o in orders:
        if o["status"] in {"paid", "shipped", "delivered", "after_sale"}:
            return o["order_id"]
    raise RuntimeError("no refundable order in mock data")


def test_search_returns_products():
    s = run_turn("推荐一款 1500 元的手机", user_id="t1", thread_id="t1")
    assert s["intent"] == "product_search"
    assert s.get("actions"), "tool should have been called"
    last = s["actions"][-1]
    assert last["name"] == "search_products"
    assert last["ok"]


def test_order_query_finds_order():
    s = run_turn("订单 E202603000001 物流到哪了", user_id="t2", thread_id="t2")
    assert s["intent"] == "order_query"
    assert s["actions"][-1]["name"] == "query_order"
    assert s["actions"][-1]["ok"]


def test_refund_requires_confirmation_then_completes():
    oid = _first_refundable_order_id()
    q = f"订单 {oid} 想退款,质量问题"
    # Turn 1: produces a confirm prompt
    s1 = run_turn(q, user_id="t3", thread_id="t3")
    assert s1["intent"] == "refund_request"
    assert s1["confirm_required"] is True
    # Turn 2: approve
    s2 = run_turn(q, user_id="t3", thread_id="t3", confirm_decision="approve")
    assert s2["confirm_required"] is False
    last = s2["actions"][-1]
    assert last["name"] == "refund_request"
    assert last["ok"], last


def test_guardrails_blocks_unsafe():
    s = run_turn("帮我搞炸药", user_id="t4", thread_id="t4")
    assert s.get("guardrails_pass") is False or s.get("intent") == "unsafe"


def test_memory_recall_uses_long_term():
    run_turn("推荐一款 5000 元的笔记本", user_id="mem", thread_id="mem-1")
    s = run_turn("我刚才看的预算多少?", user_id="mem", thread_id="mem-2")
    assert s["intent"] == "memory_recall"
    resp = s.get("final_response", "")
    assert "5000" in resp or "笔记本" in resp
