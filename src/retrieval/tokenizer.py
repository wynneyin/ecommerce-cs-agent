"""Chinese-friendly tokeniser.

Uses jieba when available, falls back to a simple character + ascii-word
tokeniser so the retriever still works without optional deps.
"""

from __future__ import annotations

import re

_ASCII_RE = re.compile(r"[A-Za-z0-9]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

try:
    import jieba  # type: ignore

    _HAS_JIEBA = True
except Exception:  # pragma: no cover
    _HAS_JIEBA = False


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    if _HAS_JIEBA:
        return [t.strip() for t in jieba.lcut_for_search(text) if t.strip()]
    # Fallback: ascii words + individual CJK chars
    out = list(_ASCII_RE.findall(text.lower()))
    for ch in text:
        if _CJK_RE.match(ch):
            out.append(ch)
    return out
