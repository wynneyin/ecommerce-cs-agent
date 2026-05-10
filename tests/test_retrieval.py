from src.retrieval import get_retriever


def test_topic_pin_pins_exact_topic():
    r = get_retriever("faq")
    res = r.retrieve("七天无理由退货怎么操作", top_k=3)
    assert res, "expected hits"
    assert res[0]["id"] == "return_window"
    assert r.last_method.startswith("topic_pin")


def test_shipping_fee_pinned():
    r = get_retriever("faq")
    res = r.retrieve("运费多少", top_k=3)
    assert res[0]["id"] == "shipping_fee"


def test_product_retrieval_returns_relevant():
    r = get_retriever("product")
    res = r.retrieve("降噪耳机", top_k=3)
    ids = [d["id"] for d in res]
    assert any("P101" in x for x in ids)
