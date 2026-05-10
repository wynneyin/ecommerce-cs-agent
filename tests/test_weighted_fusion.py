"""Weighted dual recall fusion (vector + BM25)."""

from src.retrieval.hybrid import Doc, weighted_dual_fuse


def test_weighted_fuse_favors_vector_when_configured():
    a = Doc(id="a", text="foo")
    b = Doc(id="b", text="bar")
    # a wins BM25, b wins vector
    bm = [(a, 10.0), (b, 1.0)]
    vec = [(b, 0.9), (a, 0.1)]
    out = weighted_dual_fuse(bm, vec, vec_weight=0.7, bm25_weight=0.3, pool_limit=10)
    assert len(out) == 2
    # 归一化后 b 的向量分高、a 的 BM25 高；0.7 向量权重下 b 应排第一
    assert out[0][0].id == "b"


def test_weighted_fuse_single_channel():
    only = Doc(id="x", text="z")
    out = weighted_dual_fuse([(only, 5.0)], [], vec_weight=0.7, bm25_weight=0.3, pool_limit=5)
    assert len(out) == 1
    assert out[0][0].id == "x"
