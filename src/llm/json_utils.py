"""Extract JSON objects from LLM text (markdown fences, trailing prose)."""

from __future__ import annotations

import json
import re


def parse_json_object(text: str | None) -> dict | None:
    if not text or not isinstance(text, str):
        return None
    raw = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fence:
        raw = fence.group(1).strip()
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    brace = re.search(r"\{[\s\S]*\}", raw)
    if brace:
        try:
            data = json.loads(brace.group(0))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None
    return None
