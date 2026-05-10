"""Build retrieval indexes once (so cold-start latency is amortised)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.index_builder import get_retriever  # noqa: E402


def main() -> None:
    faq = get_retriever("faq")
    prod = get_retriever("product")
    queries = ["几天发货", "无理由退货怎么操作", "推荐 5000 元的笔记本", "Echo Mini"]
    for q in queries:
        for kind, retr in (("faq", faq), ("product", prod)):
            res = retr.retrieve(q, top_k=3)
            ids = [r["id"] for r in res]
            print(f"[{kind}] {q!r} -> {ids} ({retr.last_method})")


if __name__ == "__main__":
    main()
