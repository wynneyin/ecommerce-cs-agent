"""Streamlit demo for the e-commerce customer-service agent.

* Tab「对话」: 面向用户，仅展示最终回复 + 思考中动画；可展开调试。
* Tab「Debug」: 原有完整链路 / Trace / 工具卡片（未删减）。
"""

from __future__ import annotations

import json
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
# User-facing UI: CSS + thinking animation (runs in browser during blocking work)
# ---------------------------------------------------------------------------

USER_TAB_CSS = """
<style>
    .ec-hero-wrap { text-align:center; padding: 3rem 1rem 2rem; }
    .ec-hero-title {
        font-size: clamp(1.5rem, 4vw, 2rem);
        font-weight: 600;
        color: #111;
        letter-spacing: 0.02em;
        margin-bottom: 0.5rem;
    }
    .ec-sub { color: #6b7280; font-size: 0.9rem; }
    @keyframes ec-bounce {
        0%, 80%, 100% { transform: translateY(0); opacity: 0.45; }
        40% { transform: translateY(-10px); opacity: 1; }
    }
    @keyframes ec-shimmer {
        0% { background-position: -120% center; }
        100% { background-position: 220% center; }
    }
    .ec-thinking-row {
        display: flex; align-items: center; gap: 14px;
        padding: 14px 18px; margin: 8px 0;
        border-radius: 14px;
        background: linear-gradient(90deg, #f8fafc 0%, #eef2ff 50%, #f8fafc 100%);
        background-size: 200% 100%;
        animation: ec-shimmer 2.2s ease-in-out infinite;
        border: 1px solid #e5e7eb;
    }
    .ec-dots { display: flex; gap: 6px; align-items: center; }
    .ec-dot {
        width: 9px; height: 9px; border-radius: 50%;
        background: #6366f1;
        animation: ec-bounce 0.9s ease-in-out infinite;
    }
    .ec-dot:nth-child(2) { animation-delay: 0.15s; background: #8b5cf6; }
    .ec-dot:nth-child(3) { animation-delay: 0.3s; background: #a855f7; }
    .ec-thinking-label {
        font-size: 0.95rem;
        color: #374151;
        font-weight: 500;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #9ca3af !important;
    }
</style>
"""

THINKING_HTML = (
    USER_TAB_CSS
    + """
<div class="ec-thinking-row">
  <div class="ec-dots"><div class="ec-dot"></div><div class="ec-dot"></div><div class="ec-dot"></div></div>
  <span class="ec-thinking-label">正在理解您的问题，请稍候…</span>
</div>
"""
)


# ---------------------------------------------------------------------------
# Sidebar / state
# ---------------------------------------------------------------------------


def _init_session() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())[:8]
    if "user_id" not in st.session_state:
        st.session_state.user_id = "demo"
    if "history" not in st.session_state:
        st.session_state.history = []
    if "pending_confirm" not in st.session_state:
        st.session_state.pending_confirm = None
    if "user_tab_show_debug" not in st.session_state:
        st.session_state.user_tab_show_debug = False


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
        help="fake=离线规则；openai/deepseek/ollama 走真实模型。切换后请重启应用。.env 中 LLM_PROVIDER=openai 且 USE_LLM_*=true 时为完整 NLU+思考+合成链路。",
    )
    if llm_choice != SETTINGS.llm_provider:
        os.environ["LLM_PROVIDER"] = llm_choice
    if st.button("新会话"):
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.session_state.history.clear()
        st.session_state.pending_confirm = None
        st.session_state.user_tab_show_debug = False
        st.rerun()
    st.markdown("---")
    st.markdown(
        "**对话页**仅显示助手最终回复；完整链路请切到 **Debug** 标签。"
    )
    st.markdown(
        "**指标对照 (作品集级)**\n\n"
        "Intent / Slot / Tool / Args / Recall@K / Pipeline\n"
        "→ `python scripts/run_eval.py --memory on`"
    )


# ---------------------------------------------------------------------------
# Helpers (shared with Debug tab)
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


def _render_react_flow_panel(state: dict) -> None:
    """置顶展示：拆解 → NLU → Think → Plan/Act → Reflect（ReAct 可见脉络）。"""
    mode = state.get("mode") or "-"
    with st.container(border=True):
        st.markdown("### ReAct / 模型推理脉络")
        st.caption(
            f"当前 **mode=`{mode}`** · 典型链路：**Understand（拆解）→ NLU → Retrieve? → Think → Plan → Act → Reflect → 回复**。"
            "shortcut 分支会跳过检索/Think。"
        )

        qu = (state.get("query_understanding") or "").strip()
        st.markdown("##### ① Understand · 用户问题拆解（大模型）")
        if qu:
            st.markdown(qu)
        else:
            st.caption("（无 — 未启用远程拆解或非 LLM 模式）")

        intent = state.get("intent", "-")
        ic = float(state.get("intent_conf") or 0)
        st.markdown("##### ② NLU · 意图与槽位")
        st.markdown(f"- **intent**：`{intent}`　·　**confidence**：{ic:.2f}")
        slots = state.get("slots") or {}
        if slots:
            st.markdown("- **slots**： " + " · ".join(f"`{k}`={v}" for k, v in slots.items()))
        else:
            st.caption("slots：（空）")

        think_rounds = (state.get("memory_working") or {}).get("think_rounds")
        st.markdown("##### ③ Think · 执行计划前的推理")
        if think_rounds:
            for i, t in enumerate(think_rounds, start=1):
                st.markdown(f"**第 {i} 轮**\n\n{t}")
        elif state.get("thinking"):
            st.markdown(state["thinking"])
        else:
            st.caption("（本轮未生成 Think — 可能 shortcut 或未启用）")

        plan = state.get("plan") or []
        st.markdown("##### ④ Plan → Act")
        if plan:
            st.json({"plan": plan})
        else:
            st.caption("plan：（空）")
        acts = state.get("actions") or []
        if acts:
            st.caption("工具执行摘要（最近几条）：")
            for a in acts[-5:]:
                st.markdown(
                    f"- `{a.get('name')}` · ok={a.get('ok')} · {a.get('latency_ms', 0)} ms"
                )
        else:
            st.caption("actions：（无）")

        rt = state.get("react_trace") or []
        st.markdown("##### ⑤ Reflect · 执行后反思（ReAct 闭环）")
        if rt:
            for step in rt:
                rnd = step.get("round", "?")
                need = step.get("need_replan")
                code = step.get("reflection_code", "-")
                st.markdown(
                    f"**轮次 {rnd}** · `need_replan={need}` · `{code}`"
                )
                if step.get("react_reasoning"):
                    st.markdown(step["react_reasoning"])
                else:
                    st.caption("（仅机械码，未生成反思长文）")
        else:
            st.caption("（本轮无 Reflect — shortcut / 确认分支 / deterministic 单次）")


def _stage_display_name(stage: str) -> str:
    names = {
        "query_understanding": "Understand · 用户问题拆解（自然语言）",
        "query_understanding_error": "Understand 调用异常",
        "nlu_primary": "NLU 意图识别（约定输出 JSON）",
        "nlu_refine": "NLU 兜底纠错（JSON）",
        "nlu_refine_error": "NLU 调用异常",
        "think": "Think 推理（自然语言）",
        "think_error": "Think 调用异常",
        "reflect": "Reflect 反思（自然语言）",
        "reflect_error": "Reflect 调用异常",
        "reply_synthesis": "最终话术合成（自然语言）",
        "reply_synthesis_error": "话术合成异常",
        "reply_conversational": "会话润色 / 无工具兜底（自然语言）",
        "reply_conversational_error": "会话润色异常",
    }
    return names.get(stage, stage)


def _render_llm_response_body(stage: str, resp: str) -> None:
    if not resp.strip():
        st.caption("_(空)_")
        return
    if stage.startswith("query_understanding"):
        st.caption("此为 **Understand** 步骤：内部拆解用户话，不是最终客服口吻。")
        if stage.endswith("_error"):
            st.warning(resp)
        else:
            st.markdown(resp)
        return
    if stage in ("reply_conversational", "reply_conversational_error"):
        st.caption("此为 **无工具或模板润色** 的自然语言回复。")
        if stage.endswith("_error"):
            st.warning(resp)
        else:
            st.markdown(resp)
        return
    if stage.startswith("nlu"):
        st.caption(
            "这是 **意图识别** 要求的结构化 JSON，不是给用户的聊天正文。"
            "随意输入常被模型标成 `intent=unknown`，属正常现象；最终回复请看上方气泡。"
        )
        try:
            st.json(json.loads(resp.strip()))
        except json.JSONDecodeError:
            st.code(resp, language=None)
        return
    if stage.endswith("_error"):
        st.warning(resp)
        return
    st.markdown(resp)


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
    _render_react_flow_panel(state)

    summary_list = state.get("llm_run_summary") or []
    if summary_list:
        with st.expander("🔍 本轮 LLM 调用摘录（原始输出，便于排查）", expanded=True):
            st.caption(
                "各环节含义不同：**NLU** 固定返回 JSON（intent/slots）；"
                "**Think / Reflect / 合成** 才是自然语言。"
                "对用户可见的最终话术以对话气泡为准。"
                "终端 stderr 可同步查看（`AGENT_LOG_LLM=0` 关闭）。"
            )
            for i, ex in enumerate(summary_list, start=1):
                stage = ex.get("stage", "?")
                label = _stage_display_name(stage)
                st.markdown(f"**{i}.** `{stage}` — {label}")
                hint = ex.get("prompt_hint") or ""
                if hint.strip():
                    with st.expander("prompt 摘要", expanded=False):
                        st.code(hint, language=None)
                resp = ex.get("response") or ""
                st.markdown("**模型输出：**")
                _render_llm_response_body(stage, resp)

    cols = st.columns(4)
    cols[0].metric("Intent", state.get("intent", "-"))
    cols[1].metric("Conf", f"{state.get('intent_conf', 0):.2f}")
    cols[2].metric("Retrieval", state.get("retrieval_method") or "-")
    cols[3].metric("Mode", state.get("mode") or "-")
    if state.get("nlu_timing_ms"):
        st.caption("**NLU 耗时 (ms):** " + ", ".join(f"{k}={v}" for k, v in state["nlu_timing_ms"].items()))
    if state.get("memory_long"):
        st.caption("**Memory:** " + ", ".join(f"{k}={v}" for k, v in state["memory_long"].items()))


def _render_user_tab(mode: str, use_memory: bool) -> None:
    """面向用户：仅最终回复 + 动画；可选展开调试。"""
    st.markdown(USER_TAB_CSS, unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown(
            '<div class="ec-hero-wrap"><div class="ec-hero-title">我们先从哪里开始呢？</div>'
            '<div class="ec-sub">购物、订单、售后都可以问</div></div>',
            unsafe_allow_html=True,
        )

    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        if st.button(
            "🔧 打开调试详情（本页展开）",
            use_container_width=True,
            help="展示与 Debug 标签相同的链路信息（本轮最后一条回复）",
        ):
            st.session_state.user_tab_show_debug = not st.session_state.user_tab_show_debug
            st.rerun()

    st.caption("也可直接切换到顶部 **Debug** 标签查看完整面板。")

    for entry in st.session_state.history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])

    if st.session_state.pending_confirm:
        with st.chat_message("assistant"):
            payload = st.session_state.pending_confirm[1]
            st.warning("该操作需要您确认后继续")
            st.json(payload)
            cols = st.columns(2)
            if cols[0].button("确认提交", use_container_width=True, key="user_ap"):
                user_query = st.session_state.pending_confirm[0]
                thinking_ph = st.empty()
                thinking_ph.markdown(THINKING_HTML, unsafe_allow_html=True)
                state = run_turn(
                    user_query,
                    user_id=st.session_state.user_id,
                    thread_id=st.session_state.thread_id,
                    mode=mode,
                    use_memory=use_memory,
                    confirm_decision="approve",
                )
                thinking_ph.empty()
                st.session_state.history.append(
                    {
                        "role": "assistant",
                        "content": state.get("final_response", ""),
                        "state": state,
                    }
                )
                st.session_state.pending_confirm = None
                st.rerun()
            if cols[1].button("取消", use_container_width=True, key="user_rj"):
                user_query = st.session_state.pending_confirm[0]
                thinking_ph = st.empty()
                thinking_ph.markdown(THINKING_HTML, unsafe_allow_html=True)
                state = run_turn(
                    user_query,
                    user_id=st.session_state.user_id,
                    thread_id=st.session_state.thread_id,
                    mode=mode,
                    use_memory=use_memory,
                    confirm_decision="reject",
                )
                thinking_ph.empty()
                st.session_state.history.append(
                    {
                        "role": "assistant",
                        "content": "已取消该操作。",
                        "state": state,
                    }
                )
                st.session_state.pending_confirm = None
                st.rerun()

    user_query = st.chat_input("有问题，尽管问")
    if user_query:
        st.session_state.history.append({"role": "user", "content": user_query, "state": None})
        thinking_ph = st.empty()
        thinking_ph.markdown(THINKING_HTML, unsafe_allow_html=True)
        state = run_turn(
            user_query,
            user_id=st.session_state.user_id,
            thread_id=st.session_state.thread_id,
            mode=mode,
            use_memory=use_memory,
        )
        thinking_ph.empty()
        if state.get("confirm_required"):
            st.session_state.pending_confirm = (user_query, state.get("confirm_payload"))
        st.session_state.history.append(
            {
                "role": "assistant",
                "content": state.get("final_response", ""),
                "state": state,
            }
        )
        st.rerun()

    if st.session_state.user_tab_show_debug:
        latest = next(
            (h for h in reversed(st.session_state.history) if h.get("state")),
            None,
        )
        if latest and latest.get("state"):
            with st.expander("🔧 调试详情（本轮最后一条助手回复对应的状态）", expanded=True):
                _render_state_summary(latest["state"])
        else:
            st.info("尚无带状态的回复，发一条消息后再展开。")


# ---------------------------------------------------------------------------
# Tabs: 对话 | Debug
# ---------------------------------------------------------------------------

tab_user, tab_debug = st.tabs(["对话", "Debug"])

with tab_user:
    _render_user_tab(mode, use_memory)

with tab_debug:
    col_chat, col_action, col_trace = st.columns([2, 2, 1.6], gap="large")

    with col_chat:
        st.subheader("对话")
        for entry in st.session_state.history:
            with st.chat_message(entry["role"]):
                st.markdown(entry["content"])
                if entry.get("state") and entry["role"] == "assistant":
                    _render_state_summary(entry["state"])

        if st.session_state.pending_confirm:
            with st.chat_message("assistant"):
                payload = st.session_state.pending_confirm[1]
                st.warning("检测到敏感操作,需要您确认")
                st.json(payload)
                cols = st.columns(2)
                if cols[0].button("确认提交", use_container_width=True, key="dbg_ap"):
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
                if cols[1].button("取消", use_container_width=True, key="dbg_rj"):
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

        with st.form("debug_turn_form", clear_on_submit=True):
            user_query_d = st.text_input(
                "调试发送（与对话页共用会话 thread）",
                placeholder="例如: 推荐 5000 元的笔记本 / 订单 E202603000001 物流",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("发送（Debug）")
        if submitted and user_query_d.strip():
            st.session_state.history.append(
                {"role": "user", "content": user_query_d.strip(), "state": None}
            )
            state = run_turn(
                user_query_d.strip(),
                user_id=st.session_state.user_id,
                thread_id=st.session_state.thread_id,
                mode=mode,
                use_memory=use_memory,
            )
            if state.get("confirm_required"):
                st.session_state.pending_confirm = (
                    user_query_d.strip(),
                    state.get("confirm_payload"),
                )
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
