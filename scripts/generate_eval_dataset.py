"""Generate the 120-row evaluation dataset.

Distribution (matches the resume description):
* product_search   : 20
* product_detail   : 15
* product_compare  : 15
* order_query      : 15
* refund_request   : 20
* faq_policy       : 25
* memory_recall    : 10
                    ---
TOTAL              : 120

Each row schema::

    {
        "id": "ev_001",
        "task": "product_search",
        "query": "...",
        "history": [{"role": "user"|"assistant", "content": "..."}],   # optional
        "intent": "product_search",
        "slots": {...},
        "expected_tool": "search_products",
        "expected_args": {...},
        "relevant_ids": ["P1001", ...],   # for Recall@K
        "topic": "shipping_time",         # for FAQ
    }
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

random.seed(7)


# ---------------------------------------------------------------------------
# Helpers (read mock data so relevant_ids are realistic)
# ---------------------------------------------------------------------------


def _load(name: str) -> list[dict]:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


PRODUCTS = _load("products.json")
ORDERS = _load("orders.json")
PROD_BY_CAT: dict[str, list[dict]] = {}
for p in PRODUCTS:
    PROD_BY_CAT.setdefault(p["category"], []).append(p)

CATEGORY_QUERIES = {
    "手机": ["推荐一款 {budget} 元左右的手机", "有什么 {budget} 内的手机推荐", "{budget} 元手机性价比哪家强"],
    "笔记本": [
        "想买个笔记本,预算 {budget} 元",
        "推荐一款 {budget} 元的笔记本",
        "{budget} 内办公笔记本",
    ],
    "耳机": ["{budget} 元降噪耳机推荐", "想买个 {budget} 内的耳机", "通勤用耳机推荐 {budget}"],
    "音箱": ["{budget} 内的蓝牙音箱", "推荐 {budget} 元音箱"],
    "相机": ["{budget} 元入门相机推荐", "vlog 相机预算 {budget}"],
    "平板": ["{budget} 元学生平板推荐", "买个 {budget} 内的平板"],
    "手表": ["{budget} 内运动手表", "推荐一款 {budget} 元智能手表"],
    "鞋": ["跑步鞋推荐 {budget} 内", "{budget} 元的跑步鞋"],
}


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def gen_product_search(n: int = 20) -> list[dict]:
    out: list[dict] = []
    cats = list(CATEGORY_QUERIES.keys())
    for i in range(n):
        cat = cats[i % len(cats)]
        budget = random.choice([999, 1500, 2000, 3000, 5000, 8000, 12000])
        tmpl = random.choice(CATEGORY_QUERIES[cat])
        query = tmpl.format(budget=budget)
        # relevant ids = products in this category within budget * 1.5
        relevant = [p["product_id"] for p in PROD_BY_CAT.get(cat, []) if p["price"] <= budget * 1.5]
        if not relevant:
            relevant = [p["product_id"] for p in PROD_BY_CAT.get(cat, [])][:3]
        out.append(
            {
                "id": f"search_{i:03d}",
                "task": "product_search",
                "query": query,
                "intent": "product_search",
                "slots": {"category": cat, "budget": budget},
                "expected_tool": "search_products",
                "expected_args": {"category": cat, "budget": budget},
                "relevant_ids": relevant,
            }
        )
    return out


def gen_product_detail(n: int = 15) -> list[dict]:
    out: list[dict] = []
    sampled = random.sample(PRODUCTS, k=min(n, len(PRODUCTS)))
    templates = [
        "查一下 {pid} 的详情",
        "{pid} 怎么样",
        "介绍一下 {pid}",
        "{pid} 是什么参数",
        "{pid} 这款好不好",
    ]
    for i, p in enumerate(sampled):
        pid = p["product_id"]
        query = random.choice(templates).format(pid=pid)
        out.append(
            {
                "id": f"detail_{i:03d}",
                "task": "product_detail",
                "query": query,
                "intent": "product_detail",
                "slots": {"product_id": pid},
                "expected_tool": "get_product_detail",
                "expected_args": {"product_id": pid},
                "relevant_ids": [pid],
            }
        )
    return out


def gen_product_compare(n: int = 15) -> list[dict]:
    out: list[dict] = []
    cat_with_two = [c for c, ps in PROD_BY_CAT.items() if len(ps) >= 2]
    for i in range(n):
        cat = random.choice(cat_with_two)
        a, b = random.sample(PROD_BY_CAT[cat], 2)
        templates = [
            "对比一下 {a} 和 {b}",
            "{a} 和 {b} 哪个好",
            "{a} vs {b}",
            "{a} 比 {b} 好在哪",
        ]
        query = random.choice(templates).format(a=a["product_id"], b=b["product_id"])
        out.append(
            {
                "id": f"compare_{i:03d}",
                "task": "product_compare",
                "query": query,
                "intent": "product_compare",
                "slots": {"product_ids": [a["product_id"], b["product_id"]]},
                "expected_tool": "compare_products",
                "expected_args": {"product_ids": [a["product_id"], b["product_id"]]},
                "relevant_ids": [a["product_id"], b["product_id"]],
            }
        )
    return out


def gen_order_query(n: int = 15) -> list[dict]:
    out: list[dict] = []
    sampled = random.sample(ORDERS, k=min(n, len(ORDERS)))
    templates = [
        "订单 {oid} 物流到哪了",
        "我的订单 {oid} 怎么还没到",
        "查一下订单 {oid}",
        "{oid} 这个单什么时候发货",
        "麻烦看下 {oid}",
    ]
    for i, o in enumerate(sampled):
        oid = o["order_id"]
        query = random.choice(templates).format(oid=oid)
        out.append(
            {
                "id": f"order_{i:03d}",
                "task": "order_query",
                "query": query,
                "intent": "order_query",
                "slots": {"order_id": oid},
                "expected_tool": "query_order",
                "expected_args": {"order_id": oid},
                "relevant_ids": [oid],
            }
        )
    return out


REFUND_REASONS = [
    ("质量问题", ["质量不行", "用着坏了", "破损了", "有质量问题"]),
    ("尺码不合", ["尺码不合", "尺码不对"]),
    ("不想要", ["不想要了", "不喜欢"]),
    ("发货错误", ["发错了", "发错货了"]),
    ("疑似假货", ["像是假货"]),
]


def gen_refund_request(n: int = 20) -> list[dict]:
    out: list[dict] = []
    refundable = [
        o for o in ORDERS if o["status"] in {"paid", "shipped", "delivered", "after_sale"}
    ]
    sampled = random.sample(refundable, k=min(n, len(refundable)))
    while len(sampled) < n:  # allow duplicates if not enough orders
        sampled.append(random.choice(refundable))

    templates = [
        "订单 {oid} 想退,{reason}",
        "{oid} 申请退款,{reason}",
        "{oid} 这单退了吧,{reason}",
        "想退 {oid},{reason}",
    ]
    for i, o in enumerate(sampled):
        oid = o["order_id"]
        canon, surfaces = random.choice(REFUND_REASONS)
        reason_text = random.choice(surfaces)
        query = random.choice(templates).format(oid=oid, reason=reason_text)
        out.append(
            {
                "id": f"refund_{i:03d}",
                "task": "refund_request",
                "query": query,
                "intent": "refund_request",
                "slots": {"order_id": oid, "refund_reason": canon},
                "expected_tool": "refund_request",
                "expected_args": {"order_id": oid, "reason": canon},
                "relevant_ids": [oid],
            }
        )
    return out


FAQ_QUERIES = [
    ("shipping_time", "几天发货?"),
    ("shipping_time", "下单多久能发货?"),
    ("shipping_time", "什么时候发货"),
    ("shipping_fee", "运费多少?"),
    ("shipping_fee", "包邮吗?"),
    ("return_window", "七天无理由退货怎么操作"),
    ("return_window", "无理由退货可以吗"),
    ("return_window", "可以无理由退吗"),
    ("refund_timeline", "退款几天到账?"),
    ("refund_timeline", "多久能退款"),
    ("warranty", "保修期是多久?"),
    ("warranty", "三包政策"),
    ("invoice", "怎么开发票?"),
    ("invoice", "可以开增值税专用发票吗"),
    ("membership", "会员等级有哪些"),
    ("membership", "怎么升级会员"),
    ("coupon", "优惠券怎么用?"),
    ("coupon", "优惠券可以叠加吗"),
    ("payment", "支持哪些支付方式?"),
    ("payment", "可以用花呗分期吗"),
    ("address_change", "怎么修改收货地址?"),
    ("cancel_order", "怎么取消订单?"),
    ("price_protect", "支持价保吗"),
    ("contact_support", "怎么联系人工客服?"),
    ("exchange", "怎么换货?"),
]


def gen_faq_policy(n: int = 25) -> list[dict]:
    out: list[dict] = []
    sampled = FAQ_QUERIES[:n] if len(FAQ_QUERIES) >= n else FAQ_QUERIES
    for i, (topic, query) in enumerate(sampled):
        out.append(
            {
                "id": f"faq_{i:03d}",
                "task": "faq_policy",
                "query": query,
                "intent": "faq_policy",
                "slots": {},
                "expected_tool": "faq_retrieve",
                "expected_args": {"query": query},
                "relevant_ids": [topic],
                "topic": topic,
            }
        )
    return out


def gen_memory_recall(n: int = 10) -> list[dict]:
    """Multi-turn memory cases: a setup turn + a recall turn."""
    out: list[dict] = []
    setups = [
        (
            [{"role": "user", "content": "推荐一款 5000 元的笔记本"}],
            "我刚才在看什么来着?",
            {"category": "笔记本", "budget": 5000},
        ),
        (
            [{"role": "user", "content": "对比 P1007 和 P1010"}],
            "刚才那两款笔记本哪个好?",
            {"product_ids": ["P1007", "P1010"]},
        ),
        (
            [{"role": "user", "content": "查一下 P1004 的详情"}],
            "刚说的那款手机价格呢?",
            {"product_id": "P1004"},
        ),
        (
            [{"role": "user", "content": "推荐一款 1500 元的手机"}],
            "我之前看的预算多少来着?",
            {"category": "手机", "budget": 1500},
        ),
        (
            [{"role": "user", "content": "订单 E202603000003 物流到哪了"}],
            "刚才那个订单是哪个?",
            {"order_id": "E202603000003"},
        ),
        (
            [{"role": "user", "content": "推荐 800 内的耳机"}],
            "我刚才说预算多少?",
            {"category": "耳机", "budget": 800},
        ),
        (
            [{"role": "user", "content": "推荐入门相机"}],
            "上次咨询的类目是什么?",
            {"category": "相机"},
        ),
        (
            [{"role": "user", "content": "查一下 P1018"}],
            "我刚说的那个商品 ID 是啥?",
            {"product_id": "P1018"},
        ),
        (
            [{"role": "user", "content": "推荐 3000 元的平板"}],
            "刚才说的预算?",
            {"category": "平板", "budget": 3000},
        ),
        (
            [{"role": "user", "content": "对比 P1013 和 P1014"}],
            "刚才那两款是哪两款?",
            {"product_ids": ["P1013", "P1014"]},
        ),
    ]
    for i, (history, query, recall_slots) in enumerate(setups[:n]):
        out.append(
            {
                "id": f"memory_{i:03d}",
                "task": "memory_recall",
                "query": query,
                "history": history,
                "intent": "memory_recall",
                "slots": {},
                "expected_tool": None,  # no tool, response uses memory_long
                "expected_args": {},
                "relevant_ids": [],
                "expected_recall": recall_slots,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    rows: list[dict] = []
    rows.extend(gen_product_search(20))
    rows.extend(gen_product_detail(15))
    rows.extend(gen_product_compare(15))
    rows.extend(gen_order_query(15))
    rows.extend(gen_refund_request(20))
    rows.extend(gen_faq_policy(25))
    rows.extend(gen_memory_recall(10))

    out_path = DATA / "eval_dataset.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_task: dict[str, int] = {}
    for r in rows:
        by_task[r["task"]] = by_task.get(r["task"], 0) + 1
    print(f"Wrote {len(rows)} rows -> {out_path}")
    for k, v in by_task.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
