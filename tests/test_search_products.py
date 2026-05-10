"""Regression: Chinese queries must not return empty when catalogue has matches."""

from src.tools.search_products import search_products


def test_laptop_query_finds_notebooks_with_jieba_tokens():
    out = search_products(query="我要一个笔记本", category=None, top_k=5)
    assert out["items"], "expected laptops; was failing on whitespace-only tokenization"
    assert all(x["category"] == "笔记本" for x in out["items"])


def test_laptop_query_with_category_filter():
    out = search_products(query="随便看看", category="笔记本", top_k=3)
    assert len(out["items"]) == 3
    assert all(x["category"] == "笔记本" for x in out["items"])


def test_category_alias_laptop_computer():
    out = search_products(query="办公", category="笔记本电脑", top_k=5)
    assert out["items"]
    assert all(x["category"] == "笔记本" for x in out["items"])


def test_computer_synonym_infers_laptop_category():
    """「电脑」等泛词命中检索标签；不要求型号名含「电脑」（可与平板同属泛词「电脑」）。"""
    out = search_products(query="我想买台电脑办公", category=None, top_k=5)
    assert out["items"]
    assert all(x["category"] in ("笔记本", "平板") for x in out["items"])
    assert any(x["category"] == "笔记本" for x in out["items"])


def test_seed_product_ids_boosts_despite_weak_query():
    """混合检索命中的 id 应进入结果，即使用户表述与关键词重叠弱。"""
    out = search_products(
        query="xyz 无意义",
        category=None,
        top_k=3,
        seed_product_ids=["P1006", "P1007"],
    )
    ids = {x["product_id"] for x in out["items"]}
    assert "P1006" in ids or "P1007" in ids
