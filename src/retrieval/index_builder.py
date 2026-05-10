"""Build retriever instances (FAQ + product) from the mock dataset."""

from __future__ import annotations

from functools import lru_cache

from src.llm.factory import get_embeddings
from src.retrieval.hybrid import Doc, HybridRetriever
from src.tools.data_loader import load_faq, load_products


def _faq_docs() -> list[Doc]:
    out: list[Doc] = []
    for item in load_faq():
        text = f"{item['title']} {' '.join(item.get('keywords', []))} {item['content']}"
        out.append(
            Doc(
                id=item["topic"],
                text=text,
                metadata={
                    "topic": item["topic"],
                    "title": item["title"],
                    "keywords": item.get("keywords", []),
                    "source": "faq",
                },
            )
        )
    return out


def _product_docs() -> list[Doc]:
    out: list[Doc] = []
    for p in load_products():
        text = (
            f"{p['name']} {p['category']} {' '.join(map(str, p.get('tags', [])))} "
            f"{' '.join(map(str, p.get('specs', [])))} {p.get('description', '')}"
        )
        out.append(
            Doc(
                id=p["product_id"],
                text=text,
                metadata={
                    "name": p["name"],
                    "category": p["category"],
                    "price": p["price"],
                    "tags": p.get("tags", []),
                    "source": "product",
                },
            )
        )
    return out


@lru_cache(maxsize=8)
def get_retriever(kind: str = "faq") -> HybridRetriever:
    embeddings = get_embeddings()
    if kind == "faq":
        return HybridRetriever(docs=_faq_docs(), embeddings=embeddings, use_topic_pin=True)
    if kind == "product":
        return HybridRetriever(docs=_product_docs(), embeddings=embeddings, use_topic_pin=False)
    raise ValueError(f"Unknown retriever kind: {kind}")
