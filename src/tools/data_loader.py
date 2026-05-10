"""Lazy loaders for the JSON / Markdown mock dataset."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from src.config import DATA_DIR
from src.tools.product_taxonomy import enrich_product_record


@lru_cache(maxsize=1)
def load_products() -> list[dict]:
    path = DATA_DIR / "products.json"
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [enrich_product_record(dict(p)) for p in raw]


@lru_cache(maxsize=1)
def load_orders() -> list[dict]:
    path = DATA_DIR / "orders.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def index_products() -> dict[str, dict]:
    return {p["product_id"]: p for p in load_products()}


@lru_cache(maxsize=1)
def index_orders() -> dict[str, dict]:
    return {o["order_id"]: o for o in load_orders()}


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def parse_faq_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(text)
    meta: dict = {}
    body = text
    if m:
        head, body = m.group(1), m.group(2)
        for line in head.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                if v.startswith("[") and v.endswith("]"):
                    items = [x.strip() for x in v[1:-1].split(",") if x.strip()]
                    meta[k] = items
                else:
                    meta[k] = v
    return {
        "topic": meta.get("topic", path.stem),
        "title": meta.get("title", path.stem),
        "keywords": meta.get("keywords", []) or [],
        "content": body.strip(),
        "path": str(path),
    }


@lru_cache(maxsize=1)
def load_faq() -> list[dict]:
    faq_dir = DATA_DIR / "faq"
    if not faq_dir.exists():
        return []
    return [parse_faq_file(p) for p in sorted(faq_dir.glob("*.md"))]
