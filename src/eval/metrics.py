"""Per-row + aggregate evaluation metrics.

All metrics return floats in [0, 1]; aggregation is the macro mean.

Metric definitions
------------------
* **Intent Accuracy**         pred_intent == gold_intent
* **Slot Accuracy**           field-level F1 (gold_slot field-by-field), averaged over rows
* **Effective Tool Coverage** gold tool appears at least once in `actions`
* **Effective Args Accuracy** when gold tool was called, do the args match (subset)?
* **Recall@K**                |retrieved_top_k ∩ relevant_ids| / |relevant_ids|
* **Pipeline Success**        all of: intent OK + tool OK + args OK + retrieval OK
                               + final response is non-fallback
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Per-row metrics
# ---------------------------------------------------------------------------


def intent_correct(gold: dict, pred_state: dict) -> bool:
    return (gold.get("intent") or "") == (pred_state.get("intent") or "")


def slot_f1(gold: dict, pred_state: dict) -> float:
    """Field-level F1 on slot dicts.

    For ``memory_recall`` tasks we score by ``expected_recall`` content (since
    slots are intentionally empty by design).
    """
    if gold.get("task") == "memory_recall":
        # Use response/memory consistency as a proxy for slot accuracy
        return 1.0 if memory_recall_consistent(gold, pred_state) else 0.0

    gold_slots = gold.get("slots") or {}
    pred_slots = pred_state.get("slots") or {}
    if not gold_slots and not pred_slots:
        return 1.0
    if not gold_slots:
        return 0.0 if pred_slots else 1.0

    correct = 0
    for k, v in gold_slots.items():
        pv = pred_slots.get(k)
        if isinstance(v, list):
            if isinstance(pv, list) and set(map(str, v)) == set(map(str, pv)):
                correct += 1
        else:
            if pv is not None and str(pv) == str(v):
                correct += 1

    # Only penalise *meaningful* extras — `query` is auxiliary.
    pred_relevant = {k: v for k, v in pred_slots.items() if k not in {"query"}}
    precision = correct / max(1, len(pred_relevant))
    recall = correct / max(1, len(gold_slots))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def tool_called(gold: dict, pred_state: dict) -> bool:
    expected = gold.get("expected_tool")
    if not expected:
        return True  # rows without expected tool (e.g. memory_recall)
    actions = pred_state.get("actions") or []
    return any(a.get("name") == expected for a in actions)


def args_match(gold: dict, pred_state: dict) -> float:
    """Strict superset match: do the gold expected_args appear in actual args?"""
    expected = gold.get("expected_tool")
    if not expected:
        return 1.0
    expected_args = gold.get("expected_args") or {}
    actions = pred_state.get("actions") or []
    # Prefer the action whose args best match expected (handles dry-run + confirmed pair)
    candidates = [a for a in actions if a.get("name") == expected]
    if not candidates:
        return 0.0
    if not expected_args:
        return 1.0
    best = 0.0
    for action in candidates:
        actual_args = action.get("args") or {}
        correct = 0
        for k, v in expected_args.items():
            av = actual_args.get(k)
            if isinstance(v, list):
                if isinstance(av, list) and set(map(str, v)) == set(map(str, av)):
                    correct += 1
            else:
                if av is not None and str(av) == str(v):
                    correct += 1
        score = correct / max(1, len(expected_args))
        if score > best:
            best = score
    return best


def recall_at_k(gold: dict, pred_state: dict, k: int = 5) -> float:
    relevant = set(map(str, gold.get("relevant_ids") or []))
    if not relevant:
        return 1.0  # no relevance constraint
    retrieved = pred_state.get("retrieved") or []
    actions = pred_state.get("actions") or []
    candidate_ids: list[str] = []

    # Prefer retrieved docs (FAQ / search uses retriever)
    for d in retrieved[:k]:
        candidate_ids.append(str(d.get("id")))
        # Topic-pin metadata can be useful when the doc id is the topic
        topic = d.get("metadata", {}).get("topic")
        if topic:
            candidate_ids.append(str(topic))

    # Tool outputs (search / compare / detail / order)
    for a in actions:
        out = a.get("output")
        if not isinstance(out, dict):
            continue
        if isinstance(out.get("items"), list):
            for it in out["items"][:k]:
                pid = it.get("product_id") or it.get("id")
                if pid:
                    candidate_ids.append(str(pid))
        if isinstance(out.get("item"), dict):
            pid = out["item"].get("product_id") or out["item"].get("id")
            if pid:
                candidate_ids.append(str(pid))
        if isinstance(out.get("order"), dict):
            oid = out["order"].get("order_id")
            if oid:
                candidate_ids.append(str(oid))
        if isinstance(out.get("refund_id"), str):
            candidate_ids.append(str(out.get("order_id") or ""))

    hit = relevant & set(candidate_ids)
    return len(hit) / max(1, len(relevant))


def pipeline_success(
    gold: dict,
    pred_state: dict,
    *,
    intent_ok: bool,
    tool_ok: bool,
    args_score: float,
    recall_score: float,
) -> bool:
    if not intent_ok:
        return False
    if not tool_ok:
        return False
    if args_score < 0.99:
        return False
    # Recall threshold: product_search has multiple gold-relevant items, only
    # require partial overlap (>=0.5). Other tasks have small gold sets and
    # benefit from a tighter check.
    task = gold.get("task")
    recall_threshold = 0.5 if task == "product_search" else 0.99
    if recall_score < recall_threshold:
        return False
    resp = (pred_state.get("final_response") or "").strip()
    if not resp:
        return False
    fallback_markers = ("暂未理解您的问题", "暂未找到", "请提供您要", "请检查订单号")
    if any(m in resp for m in fallback_markers):
        return False
    return True


# ---------------------------------------------------------------------------
# Memory eval
# ---------------------------------------------------------------------------


def memory_recall_consistent(gold: dict, pred_state: dict) -> bool:
    expected = gold.get("expected_recall") or {}
    if not expected:
        return True
    resp = (pred_state.get("final_response") or "")
    mem = pred_state.get("memory_long") or {}

    for k, v in expected.items():
        if isinstance(v, list):
            v_str = " ".join(map(str, v))
            mem_v = mem.get("last_product_ids")
            if mem_v and set(map(str, mem_v)) == set(map(str, v)):
                continue
            if all(str(x) in resp for x in v):
                continue
            return False
        else:
            v_str = str(v)
            if v_str in resp:
                continue
            for mk, mv in mem.items():
                if str(mv) == v_str:
                    break
            else:
                return False
    return True


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute the headline metrics across all rows.

    Each row dict must contain the per-row metric values produced by
    :func:`evaluate_row`.
    """
    if not rows:
        return {}

    def _avg(key: str) -> float:
        vals = [float(r.get(key, 0.0)) for r in rows if key in r]
        return sum(vals) / len(vals) if vals else 0.0

    def _rate(key: str) -> float:
        vals = [bool(r.get(key)) for r in rows if key in r]
        return sum(1 for v in vals if v) / len(vals) if vals else 0.0

    return {
        "intent_accuracy": _rate("intent_ok"),
        "slot_accuracy": _avg("slot_f1"),
        "effective_tool_coverage": _rate("tool_ok"),
        "effective_args_accuracy": _avg("args_score"),
        "recall_at_k": _avg("recall_score"),
        "pipeline_success_rate": _rate("pipeline_ok"),
    }


def evaluate_row(gold: dict, pred_state: dict, *, k: int = 5) -> dict[str, Any]:
    intent_ok = intent_correct(gold, pred_state)
    s_f1 = slot_f1(gold, pred_state)
    t_ok = tool_called(gold, pred_state)
    a_score = args_match(gold, pred_state)
    r_score = recall_at_k(gold, pred_state, k=k)
    p_ok = pipeline_success(
        gold,
        pred_state,
        intent_ok=intent_ok,
        tool_ok=t_ok,
        args_score=a_score,
        recall_score=r_score,
    )
    return {
        "id": gold.get("id"),
        "task": gold.get("task"),
        "intent_ok": intent_ok,
        "slot_f1": s_f1,
        "tool_ok": t_ok,
        "args_score": a_score,
        "recall_score": r_score,
        "pipeline_ok": p_ok,
    }
