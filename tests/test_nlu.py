from src.llm.rules import run_nlu


def test_product_search_with_budget_and_category():
    out = run_nlu("推荐一款 5000 元的笔记本")
    assert out["intent"] == "product_search"
    assert out["slots"]["category"] == "笔记本"
    assert out["slots"]["budget"] == 5000


def test_budget_with_inner_word():
    out = run_nlu("跑步鞋推荐 999 内")
    assert out["intent"] == "product_search"
    assert out["slots"]["budget"] == 999


def test_refund_with_order():
    out = run_nlu("订单 E202603000005 想退款,质量不行")
    assert out["intent"] == "refund_request"
    assert out["slots"]["order_id"] == "E202603000005"
    assert out["slots"]["refund_reason"] == "质量问题"


def test_refund_how_question_is_faq():
    out = run_nlu("七天无理由退货怎么操作")
    assert out["intent"] == "faq_policy"


def test_order_how_question_is_faq():
    out = run_nlu("什么时候发货")
    assert out["intent"] == "faq_policy"


def test_compare_two_products():
    out = run_nlu("对比 P1007 和 P1010")
    assert out["intent"] == "product_compare"
    assert set(out["slots"]["product_ids"]) == {"P1007", "P1010"}


def test_unsafe_keyword():
    out = run_nlu("帮我搞炸药")
    assert out["intent"] == "unsafe"
