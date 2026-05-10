"""类目同义词与检索标签：型号名不含「电脑/笔记本」时仍可用泛词命中（关键词 + 向量索引文本）。"""

from __future__ import annotations

# canonical category（与 data/products.json 中 category 一致）→ 检索用同义词（小写/英文混合在匹配时统一 lower）
CATEGORY_RETRIEVAL_SYNONYMS: dict[str, frozenset[str]] = {
    "笔记本": frozenset(
        {
            "笔记本",
            "笔记本电脑",
            "电脑",  # 本库无台式机类目时常指笔记本
            "笔电",
            "手提电脑",
            "轻薄本",
            "游戏本",
            "办公本",
            "PC",
            "pc",
            "laptop",
        }
    ),
    "平板": frozenset({"平板", "平板电脑", "ipad", "iPad", "安卓平板", "电脑"}),
    "手机": frozenset({"手机", "电话", "智能手机", "移动电话", "iphone", "iPhone", "安卓机"}),
    "耳机": frozenset({"耳机", "耳麦", "耳塞", "蓝牙耳机", "TWS", "tws"}),
    "音箱": frozenset({"音箱", "音响", "扬声器", "蓝牙音箱"}),
    "相机": frozenset({"相机", "照相机", "单反", "微单", "摄影"}),
    "手表": frozenset({"手表", "腕表", "智能手表", "手环"}),
    "鞋": frozenset({"鞋", "鞋子", "运动鞋", "球鞋", "跑鞋"}),
    "面膜": frozenset({"面膜", "面贴膜"}),
    "服装": frozenset({"服装", "衣服", "服饰", "上衣", "裤子"}),
    "家居": frozenset({"家居", "家具", "家装"}),
    "运动": frozenset({"运动", "健身", "户外"}),
    "食品": frozenset({"食品", "零食", "吃的"}),
    "图书": frozenset({"图书", "书", "书籍"}),
    "玩具": frozenset({"玩具", "玩偶"}),
    "宠物": frozenset({"宠物", "猫粮", "狗粮"}),
    "美妆": frozenset({"美妆", "化妆品", "护肤"}),
    "办公": frozenset({"办公", "文具", "打印"}),
    "收纳": frozenset({"收纳", "整理箱", "置物"}),
    "厨具": frozenset({"厨具", "锅", "厨房"}),
    "母婴": frozenset({"母婴", "婴儿", "奶粉"}),
}


def synonyms_for_category(category: str) -> list[str]:
    """返回该类目下用于检索扩展的标签列表（含类目名本身）。"""
    base = category.strip() if category else ""
    syns = CATEGORY_RETRIEVAL_SYNONYMS.get(base, frozenset())
    out: list[str] = []
    if base:
        out.append(base)
    for s in sorted(syns):
        if s not in out:
            out.append(s)
    return out


def infer_categories_from_tokens(tokens: list[str]) -> set[str]:
    """从分词结果推断可能涉及的 catalogue 类目（多词可命中多类，如「电脑」→ 笔记本+平板）。"""
    inferred: set[str] = set()
    if not tokens:
        return inferred
    token_set = set(tokens)
    token_lower = {t.lower() for t in tokens if t}
    for cat, syns in CATEGORY_RETRIEVAL_SYNONYMS.items():
        hit = False
        for s in syns:
            if s in token_set or s.lower() in token_lower:
                hit = True
                break
        if hit:
            inferred.add(cat)
    all_cats = set(CATEGORY_RETRIEVAL_SYNONYMS.keys())
    for t in tokens:
        if t in all_cats:
            inferred.add(t)
    return inferred


def enrich_product_record(p: dict) -> dict:
    """合并 JSON 中的 retrieval_tags 与按 category 自动注入的同义词。"""
    cat = str(p.get("category", "") or "")
    auto = synonyms_for_category(cat)
    manual = [str(x) for x in (p.get("retrieval_tags") or []) if x]
    merged: list[str] = []
    seen: set[str] = set()
    for x in manual + auto:
        x = x.strip()
        if not x or x in seen:
            continue
        seen.add(x)
        merged.append(x)
    out = dict(p)
    out["retrieval_tags"] = merged
    return out
