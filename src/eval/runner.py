"""End-to-end evaluation runner.

Reads ``data/eval_dataset.jsonl``, runs each row through the agent, computes
metrics and writes a JSON + Markdown report under ``reports/``.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from src.config import REPORTS_DIR
from src.eval.metrics import aggregate_metrics, evaluate_row, memory_recall_consistent
from src.graph import run_turn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_dataset(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _by_task(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["task"], []).append(r)
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_eval(
    dataset_path: Path,
    *,
    mode: str = "deterministic",
    use_memory: bool = True,
    top_k: int = 5,
) -> dict:
    rows = _read_dataset(dataset_path)
    per_row: list[dict] = []
    raw_traces: list[dict] = []
    memory_eval: list[dict] = []
    t0 = time.perf_counter()

    for idx, gold in enumerate(rows):
        thread_id = f"eval-{gold['id']}"
        # Replay history first (if any) so memory_long / checkpoint accumulates
        history = gold.get("history") or []
        for turn in history:
            if turn.get("role") == "user":
                run_turn(
                    turn["content"],
                    user_id=thread_id,
                    thread_id=thread_id,
                    mode=mode,
                    use_memory=use_memory,
                )
        # Final query
        state = run_turn(
            gold["query"],
            user_id=thread_id,
            thread_id=thread_id,
            mode=mode,
            use_memory=use_memory,
        )
        # If the agent paused on a sensitive op, simulate the user approving so
        # we exercise the full flow during evaluation.
        if state.get("confirm_required"):
            state = run_turn(
                gold["query"],
                user_id=thread_id,
                thread_id=thread_id,
                mode=mode,
                use_memory=use_memory,
                confirm_decision="approve",
            )
        row_metrics = evaluate_row(gold, state, k=top_k)
        per_row.append(row_metrics)

        if gold.get("task") == "memory_recall":
            memory_eval.append(
                {
                    "id": gold["id"],
                    "consistent": memory_recall_consistent(gold, state),
                }
            )

        raw_traces.append(
            {
                "id": gold["id"],
                "query": gold["query"],
                "intent": state.get("intent"),
                "slots": state.get("slots"),
                "actions": [
                    {"name": a.get("name"), "args": a.get("args"), "ok": a.get("ok")}
                    for a in (state.get("actions") or [])
                ],
                "final_response": state.get("final_response"),
                "retrieval_method": state.get("retrieval_method"),
            }
        )

    overall = aggregate_metrics(per_row)
    by_task = {}
    for task, group in _by_task(per_row).items():
        by_task[task] = aggregate_metrics(group)

    elapsed = time.perf_counter() - t0
    return {
        "config": {
            "mode": mode,
            "use_memory": use_memory,
            "top_k": top_k,
            "n_rows": len(rows),
        },
        "elapsed_sec": round(elapsed, 2),
        "overall": overall,
        "by_task": by_task,
        "memory_eval": memory_eval,
        "rows": per_row,
        "traces": raw_traces,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.2f}%"


def write_reports(results: dict, *, prefix: str = "eval") -> tuple[Path, Path]:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = REPORTS_DIR / f"{prefix}_{ts}.json"
    md_path = REPORTS_DIR / f"{prefix}_{ts}.md"

    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    overall = results["overall"]
    cfg = results["config"]

    md_lines: list[str] = []
    md_lines.append(f"# Evaluation Report ({prefix})")
    md_lines.append("")
    md_lines.append(
        f"- mode = `{cfg['mode']}`, memory = `{cfg['use_memory']}`, "
        f"top_k = `{cfg['top_k']}`, n = `{cfg['n_rows']}`"
    )
    md_lines.append(f"- elapsed: {results['elapsed_sec']} s")
    md_lines.append("")
    md_lines.append("## Headline metrics")
    md_lines.append("")
    md_lines.append("| Metric | Value |")
    md_lines.append("|---|---|")
    md_lines.append(f"| Intent Accuracy | {_fmt_pct(overall.get('intent_accuracy'))} |")
    md_lines.append(f"| Slot Accuracy | {_fmt_pct(overall.get('slot_accuracy'))} |")
    md_lines.append(
        f"| Effective Tool Coverage | {_fmt_pct(overall.get('effective_tool_coverage'))} |"
    )
    md_lines.append(
        f"| Effective Args Accuracy | {_fmt_pct(overall.get('effective_args_accuracy'))} |"
    )
    md_lines.append(f"| Recall@K | {_fmt_pct(overall.get('recall_at_k'))} |")
    md_lines.append(
        f"| Pipeline Success Rate | {_fmt_pct(overall.get('pipeline_success_rate'))} |"
    )
    md_lines.append("")
    md_lines.append("## By-task breakdown")
    md_lines.append("")
    md_lines.append(
        "| Task | Intent | Slot | Tool | Args | Recall@K | Pipeline |"
    )
    md_lines.append("|---|---|---|---|---|---|---|")
    for task, m in results["by_task"].items():
        md_lines.append(
            "| {task} | {ia} | {sl} | {tc} | {aa} | {rk} | {pl} |".format(
                task=task,
                ia=_fmt_pct(m.get("intent_accuracy")),
                sl=_fmt_pct(m.get("slot_accuracy")),
                tc=_fmt_pct(m.get("effective_tool_coverage")),
                aa=_fmt_pct(m.get("effective_args_accuracy")),
                rk=_fmt_pct(m.get("recall_at_k")),
                pl=_fmt_pct(m.get("pipeline_success_rate")),
            )
        )

    if results.get("memory_eval"):
        n = len(results["memory_eval"])
        ok = sum(1 for m in results["memory_eval"] if m["consistent"])
        md_lines.append("")
        md_lines.append("## Memory consistency (memory_recall task)")
        md_lines.append("")
        md_lines.append(f"- consistent rate: {_fmt_pct(ok / n) if n else '—'} ({ok}/{n})")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return json_path, md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent evaluation")
    parser.add_argument(
        "--dataset",
        default="data/eval_dataset.jsonl",
        help="path to JSONL dataset",
    )
    parser.add_argument("--mode", default="deterministic", choices=["deterministic", "react"])
    parser.add_argument("--memory", default="on", choices=["on", "off"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--prefix", default="eval")
    args = parser.parse_args()

    use_memory = args.memory == "on"
    results = run_eval(
        Path(args.dataset),
        mode=args.mode,
        use_memory=use_memory,
        top_k=args.top_k,
    )
    json_path, md_path = write_reports(results, prefix=args.prefix)
    overall = results["overall"]
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print("Overall:")
    for k, v in overall.items():
        print(f"  {k}: {_fmt_pct(v)}")


if __name__ == "__main__":
    main()
