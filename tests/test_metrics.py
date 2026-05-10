from src.eval.metrics import (
    aggregate_metrics,
    evaluate_row,
    intent_correct,
    recall_at_k,
    slot_f1,
)


def test_intent_correct():
    gold = {"intent": "product_search"}
    assert intent_correct(gold, {"intent": "product_search"})
    assert not intent_correct(gold, {"intent": "order_query"})


def test_slot_f1_perfect():
    gold = {"slots": {"category": "手机", "budget": 1500}}
    pred = {"slots": {"category": "手机", "budget": 1500}}
    assert slot_f1(gold, pred) == 1.0


def test_slot_f1_partial():
    gold = {"slots": {"category": "手机", "budget": 1500}}
    pred = {"slots": {"category": "手机"}}
    f = slot_f1(gold, pred)
    assert 0.0 < f < 1.0


def test_recall_at_k_full_hit():
    gold = {"relevant_ids": ["P1004"]}
    pred = {
        "actions": [
            {
                "name": "get_product_detail",
                "output": {"item": {"product_id": "P1004"}},
            }
        ]
    }
    assert recall_at_k(gold, pred) == 1.0


def test_aggregate_metrics():
    rows = [
        {"intent_ok": True, "slot_f1": 1.0, "tool_ok": True, "args_score": 1.0, "recall_score": 1.0, "pipeline_ok": True},
        {"intent_ok": False, "slot_f1": 0.5, "tool_ok": True, "args_score": 0.5, "recall_score": 1.0, "pipeline_ok": False},
    ]
    agg = aggregate_metrics(rows)
    assert agg["intent_accuracy"] == 0.5
    assert 0.7 <= agg["slot_accuracy"] <= 0.8


def test_evaluate_row_basic():
    gold = {
        "id": "x",
        "task": "product_detail",
        "intent": "product_detail",
        "slots": {"product_id": "P1004"},
        "expected_tool": "get_product_detail",
        "expected_args": {"product_id": "P1004"},
        "relevant_ids": ["P1004"],
    }
    pred = {
        "intent": "product_detail",
        "slots": {"product_id": "P1004"},
        "actions": [
            {"name": "get_product_detail", "args": {"product_id": "P1004"}, "output": {"item": {"product_id": "P1004"}}, "ok": True}
        ],
        "final_response": "P1004 详情……",
    }
    r = evaluate_row(gold, pred)
    assert r["intent_ok"]
    assert r["tool_ok"]
    assert r["args_score"] == 1.0
    assert r["recall_score"] == 1.0
    assert r["pipeline_ok"]
