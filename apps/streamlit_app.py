"""Streamlit demo for the e-commerce customer-service agent.

Usage::

    streamlit run apps/streamlit_app.py

Layout (3 columns):
* Left   — chat (user / assistant turns)
* Middle — last action result rendered as a card (product / order / refund)
* Right  — collapsible trace tree (per-node start/end with latency)
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SETTINGS  # noqa: E402
from src.graph import run_turn  # noqa: E402

st.set_page_config(page_title="电商客服 Agent", layout="wide")


# ---------------------------------------------------------------------------
# Sidebar / state
# ---------------------------------------------------------------------------


def _init_session() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())[:8]
    if "user_id" not in st.session_state:
        st.session_state.user_id = "demo"
    if "history" not in st.session_state:
        st.session_state.history = []  # [{"role": str, "content": str, "state": dict|None}]
    if "pending_confirm" not in st.session_state:
        st.session_state.pending_confirm = None  # tuple (user_query, payload)


_init_session()


with st.sidebar:
    st.title("电商客服 Agent")
    st.caption("LangGraph · 双模执行 · 可观测 Trace")
    mode = st.radio("执行模式", ["deterministic", "react"], horizontal=True, index=0)
    use_memory = st.toggle("启用多层级记忆", value=True)
    llm_choice = st.selectbox(
        "LLM Provider",
        ["fake", "openai", "deepseek", "ollama"],
        index=["fake", "openai", "deepseek", "ollama"].index(SETTINGS.llm_provider)
        if SETTINGS.llm_provider in ("fake", "openai", "deepseek", "ollama")
        else 0,
        help="切换后请重启应用使其生效;fake 模式无需 API key 即可演示。",
    )
    if llm_choice != SETTINGS.llm_provider:
        os.environ["LLM_PROVIDER"] = llm_choice
    if st.button("新会话"):
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.session_state.history.clear()
        st.session_state.pending_confirm = None
        st.rerun()
    st.markdown("---")
    st.markdown(
        "**指标对照 (作品集级)**\n\n"
        "Intent / Slot / Tool / Args / Recall@K / Pipeline\n"
        "→ `python scripts/run_eval.py --memory on`"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _latest_action(state: dict) -> dict | None:
    actions = state.get("actions") or []
    if not actions:
        return None
    return actions[-1]


def _render_action_card(action: dict | None) -> None:
    if not action:
        st.info("暂无工具调用")
        return
    name = action.get("name")
    out = action.get("output") if isinstance(action.get("output"), dict) else {}
    st.markdown(f"**工具:** `{name}`  ·  耗时 {action.get('latency_ms', 0)} ms")

    if name == "search_products" and out.get("items"):
        for item in out["items"][:5]:
            with st.container(border=True):
                cols = st.columns([3, 1])
                cols[0].markdown(f"**{item['name']}** · `{item['product_id']}`")
                cols[0].caption(
                    f"分类 {item['category']}  |  评分 {item['rating']}  |  标签 {', '.join(item.get('tags', []))}"
                )
                cols[1].metric("价格", f"¥{item['price']}")
        st.caption(f"共 {out.get('total', 0)} 条匹配")
        return

    if name == "get_product_detail" and isinstance(out.get("item"), dict):
        it = out["item"]
        with st.container(border=True):
            st.markdown(f"### {it['name']}  ·  `{it['product_id']}`")
            cols = st.columns(3)
            cols[0].metric("价格", f"¥{it['price']}")
            cols[1].metric("评分", it["rating"])
            cols[2].metric("库存", it.get("stock", "-"))
            st.markdown("**卖点:** " + ", ".join(it.get("tags", [])))
            st.markdown("**参数:** " + ", ".join(map(str, it.get("specs", []))))
        return

    if name == "compare_products" and out.get("items"):
        items = out["items"]
        st.markdown(f"**对比 {len(items)} 款商品**")
        keys = ["name", "category", "price", "rating", "tags", "specs"]
        rows = []
        for it in items:
            rows.append(
                {
                    "name": it["name"],
                    "category": it["category"],
                    "price": it["price"],
                    "rating": it["rating"],
                    "tags": ", ".join(it.get("tags", [])),
                    "specs": ", ".join(map(str, it.get("specs", []))),
                }
            )
        st.dataframe(rows, use_container_width=True)
        st.success(out.get("summary", ""))
        return

    if name == "query_order" and isinstance(out.get("order"), dict):
        order = out["order"]
        with st.container(border=True):
            st.markdown(f"### 订单 `{order['order_id']}`")
            cols = st.columns(2)
            cols[0].markdown(f"**商品:** {order['product_name']} × {order['quantity']}")
            cols[0].markdown(f"**金额:** ¥{order['amount']}")
            cols[1].markdown(f"**状态:** {order.get('status_label', order['status'])}")
            cols[1].markdown(f"**下单日:** {order['created_at']}")
            if order.get("courier"):
                st.caption(f"物流 {order['courier']}  ·  {order['tracking_no']}")
            st.caption(f"地址: {order.get('address', '-')}")
        return

    if name == "refund_request":
        if out.get("ok") and out.get("refund_id"):
            st.success(out.get("message"))
        else:
            preview = out.get("preview") or {}
            st.warning("退款需要确认")
            st.json(preview)
        return

    if name == "faq_retrieve" and out.get("items"):
        for d in out["items"][:3]:
            with st.expander(f"{d.get('metadata', {}).get('title', d.get('id'))}"):
                st.caption(f"score = {d.get('score', 0):.4f}  ·  source = {d.get('source')}")
                st.write(d.get("content", "")[:500])
        return

    st.json(out)


def _render_trace(events: list[dict]) -> None:
    if not events:
        st.caption("(empty)")
        return
    for ev in events:
        phase = ev.get("phase")
        node = ev.get("node")
        lat = ev.get("latency_ms")
        marker = {"start": "▶", "end": "■", "error": "✖"}.get(phase, "·")
        title = f"{marker} `{node}`"
        if phase == "end" and lat:
            title += f"  · {lat:.1f} ms"
        with st.expander(title, expanded=False):
            st.json(ev.get("payload"))


def _render_state_summary(state: dict) -> None:
    cols = st.columns(4)
    cols[0].metric("Intent", state.get("intent", "-"))
    cols[1].metric("Conf", f"{state.get('intent_conf', 0):.2f}")
    cols[2].metric("Retrieval", state.get("retrieval_method") or "-")
    cols[3].metric("Mode", state.get("mode") or "-")
    if state.get("slots"):
        st.caption("**Slots:** " + ", ".join(f"{k}={v}" for k, v in state["slots"].items()))
    if state.get("memory_long"):
        st.caption("**Memory:** " + ", ".join(f"{k}={v}" for k, v in state["memory_long"].items()))


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


col_chat, col_action, col_trace = st.columns([2, 2, 1.6], gap="large")


# Chat history
with col_chat:
    st.subheader("对话")
    for entry in st.session_state.history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry.get("state") and entry["role"] == "assistant":
                _render_state_summary(entry["state"])

    # Confirm button if pending
    if st.session_state.pending_confirm:
        with st.chat_message("assistant"):
            payload = st.session_state.pending_confirm[1]
            st.warning("检测到敏感操作,需要您确认")
            st.json(payload)
            cols = st.columns(2)
            if cols[0].button("确认提交", use_container_width=True):
                user_query = st.session_state.pending_confirm[0]
                state = run_turn(
                    user_query,
                    user_id=st.session_state.user_id,
                    thread_id=st.session_state.thread_id,
                    mode=mode,
                    use_memory=use_memory,
                    confirm_decision="approve",
                )
                st.session_state.history.append(
                    {
                        "role": "assistant",
                        "content": state.get("final_response", ""),
                        "state": state,
                    }
                )
                st.session_state.pending_confirm = None
                st.rerun()
            if cols[1].button("取消", use_container_width=True):
                user_query = st.session_state.pending_confirm[0]
                state = run_turn(
                    user_query,
                    user_id=st.session_state.user_id,
                    thread_id=st.session_state.thread_id,
                    mode=mode,
                    use_memory=use_memory,
                    confirm_decision="reject",
                )
                st.session_state.history.append(
                    {
                        "role": "assistant",
                        "content": "已取消该操作。",
                        "state": state,
                    }
                )
                st.session_state.pending_confirm = None
                st.rerun()

    user_query = st.chat_input("请输入您的问题(例如: 推荐 5000 元的笔记本 / 订单 E202603000001 物流)")
    if user_query:
        st.session_state.history.append({"role": "user", "content": user_query, "state": None})
        state = run_turn(
            user_query,
            user_id=st.session_state.user_id,
            thread_id=st.session_state.thread_id,
            mode=mode,
            use_memory=use_memory,
        )
        if state.get("confirm_required"):
            st.session_state.pending_confirm = (user_query, state.get("confirm_payload"))
            st.session_state.history.append(
                {
                    "role": "assistant",
                    "content": state.get("final_response", ""),
                    "state": state,
                }
            )
        else:
            st.session_state.history.append(
                {
                    "role": "assistant",
                    "content": state.get("final_response", ""),
                    "state": state,
                }
            )
        st.rerun()


with col_action:
    st.subheader("工具结果")
    if st.session_state.history:
        latest = next(
            (h for h in reversed(st.session_state.history) if h.get("state")),
            None,
        )
        if latest and latest.get("state"):
            _render_action_card(_latest_action(latest["state"]))
        else:
            st.info("暂无")
    else:
        st.info("等待用户输入…")


with col_trace:
    st.subheader("Trace")
    if st.session_state.history:
        latest = next(
            (h for h in reversed(st.session_state.history) if h.get("state")),
            None,
        )
        if latest and latest.get("state"):
            _render_trace(latest["state"].get("trace") or [])
        else:
            st.caption("(empty)")
    else:
        st.caption("(empty)")
