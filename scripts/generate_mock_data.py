"""Generate deterministic mock e-commerce data for the agent.

Outputs:
* ``data/products.json`` (~50 SKUs across 8 categories)
* ``data/orders.json`` (~30 orders referencing the products)
* ``data/faq/*.md`` (15 policy / FAQ documents with `topic`+`keywords` front-matter)

Run:

    python scripts/generate_mock_data.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FAQ = DATA / "faq"
DATA.mkdir(exist_ok=True)
FAQ.mkdir(exist_ok=True)

random.seed(42)


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------

CATEGORIES = {
    "手机": [
        ("星河 Pro 14", 5499, ["6.7寸", "5000mAh", "1TB"], ["旗舰", "拍照"]),
        ("星河 Lite", 1999, ["6.5寸", "4500mAh", "128GB"], ["性价比", "学生"]),
        ("银河 Ultra", 7999, ["2K曲面", "潜望长焦", "1TB"], ["旗舰", "商务"]),
        ("初芒 X2", 1299, ["6.1寸", "5000mAh", "64GB"], ["入门", "百元机"]),
        ("初芒 Note", 1799, ["6.7寸", "5500mAh", "256GB"], ["大屏", "续航"]),
        ("Phoenix S25", 4399, ["钛合金", "AI助手", "512GB"], ["旗舰", "轻薄"]),
    ],
    "笔记本": [
        ("银翼 Air 14", 6499, ["i5-13", "16G/512G", "2.8K"], ["办公", "学生"]),
        ("银翼 Pro 16", 11999, ["i9-13", "32G/1T", "RTX4060"], ["设计", "游戏"]),
        ("墨竹 X14 Plus", 5499, ["R7-7840", "16G/1T", "OLED"], ["性价比", "设计"]),
        ("墨竹 Y9000K", 14999, ["i9HX", "32G/2T", "RTX4080"], ["旗舰", "游戏"]),
        ("MaiBook Air 13", 8999, ["M3", "16G/512G", "13寸"], ["轻薄", "办公"]),
        ("MaiBook Pro 14", 14999, ["M3 Pro", "18G/1T", "ProMotion"], ["旗舰", "创作"]),
    ],
    "耳机": [
        ("声波 Pods 3", 1299, ["主动降噪", "30h", "蓝牙5.3"], ["降噪", "通勤"]),
        ("声波 Pods Lite", 399, ["半入耳", "24h", "通话降噪"], ["性价比", "学生"]),
        ("Voyager Q35", 2499, ["Hi-Res", "40h", "头戴"], ["发烧", "高保真"]),
        ("Boom 100", 199, ["运动入耳", "8h", "IPX5"], ["运动", "百元"]),
    ],
    "音箱": [
        ("Echo Mini", 299, ["蓝牙", "10h", "桌面"], ["桌面", "便携"]),
        ("Echo Boom", 999, ["立体声", "防水", "30W"], ["户外", "派对"]),
        ("Hifi Star", 4999, ["书架", "85W", "无线"], ["发烧", "客厅"]),
    ],
    "相机": [
        ("光影 A7C2", 13999, ["全画幅", "33MP", "4K60"], ["全画幅", "视频"]),
        ("光影 ZV1F", 4799, ["1英寸", "Vlog", "4K"], ["Vlog", "便携"]),
        ("快拍 X100Vi", 12999, ["复古", "定焦", "胶片模拟"], ["复古", "街拍"]),
    ],
    "平板": [
        ("漫游 Pad 11", 2299, ["120Hz", "8200mAh", "256G"], ["性价比", "网课"]),
        ("漫游 Pad Pro", 5499, ["MiniLED", "M2同款", "1T"], ["旗舰", "生产力"]),
        ("MaiPad Air 11", 4799, ["M2", "11寸", "256G"], ["轻薄", "学生"]),
    ],
    "手表": [
        ("健行 Watch S3", 1499, ["GPS", "心率", "14天续航"], ["运动", "健康"]),
        ("健行 Watch GT", 999, ["血氧", "10天", "圆表"], ["性价比", "健康"]),
        ("MaiWatch S10", 2999, ["蜂窝版", "ECG", "钛壳"], ["旗舰", "健康"]),
    ],
    "鞋": [
        ("飞行 Run 5", 599, ["缓震", "透气", "马拉松"], ["跑步", "缓震"]),
        ("飞行 Trail 2", 799, ["越野", "防滑", "防水"], ["越野", "户外"]),
        ("国潮 Old 1", 399, ["复古", "板鞋", "百搭"], ["板鞋", "百搭"]),
    ],
}


def build_products() -> list[dict]:
    out: list[dict] = []
    pid = 1000
    for cat, items in CATEGORIES.items():
        for name, price, specs, tags in items:
            pid += 1
            stock = random.randint(0, 80)
            rating = round(random.uniform(4.2, 4.95), 2)
            out.append(
                {
                    "product_id": f"P{pid}",
                    "name": name,
                    "category": cat,
                    "price": price,
                    "stock": stock,
                    "rating": rating,
                    "specs": specs,
                    "tags": tags,
                    "description": f"{name} 是一款主打{tags[0]}的{cat},核心卖点:{', '.join(specs)}。",
                }
            )
    # extras to reach ~50 SKUs (plan target)
    extras = [
        ("face", "面膜", "肌研 玻尿酸面膜 5片", 89, ["补水", "敏感肌"]),
        ("face", "面膜", "雪域 修护安瓶 30ml", 199, ["修护", "夜用"]),
        ("kid", "服装", "童趣 卡通短袖", 79, ["儿童", "纯棉"]),
        ("kid", "服装", "童趣 防晒外套", 159, ["UPF50+", "夏季"]),
        ("home", "家居", "云眠 记忆枕 单只", 129, ["睡眠", "护颈"]),
        ("home", "家居", "清尘 无线吸尘器 Lite", 899, ["清洁", "入门"]),
        ("home", "家居", "小厨 空气炸锅 4L", 399, ["厨房", "少油"]),
        ("sport", "运动", "劲跑 跳绳 计数款", 59, ["健身", "有氧"]),
        ("sport", "运动", "山行 登山杖 碳纤维", 199, ["户外", "徒步"]),
        ("food", "食品", "每日坚果 混合装 500g", 79, ["零食", "礼盒"]),
        ("food", "食品", "有机燕麦片 1kg", 45, ["早餐", "健康"]),
        ("book", "图书", "Python 入门到实践 第3版", 89, ["编程", "教材"]),
        ("toy", "玩具", "积木城市系列 街景", 299, ["儿童", "益智"]),
        ("pet", "宠物", "宠悦 冻干猫粮 2kg", 159, ["猫粮", "冻干"]),
        ("beauty", "美妆", "丝绒雾面口红 豆沙色", 129, ["口红", "显白"]),
        ("office", "办公", "静音无线鼠标 M200", 79, ["办公", "静音"]),
        ("storage", "收纳", "透明抽屉式收纳柜 5层", 199, ["收纳", "宿舍"]),
        ("kitchen", "厨具", "不锈钢炒锅 32cm", 249, ["不粘", "燃气通用"]),
        ("baby", "母婴", "婴儿湿巾 80抽×6包", 49, ["婴儿", "无香"]),
    ]
    for _, cat, name, price, tags in extras:
        pid += 1
        out.append(
            {
                "product_id": f"P{pid}",
                "name": name,
                "category": cat,
                "price": price,
                "stock": random.randint(20, 200),
                "rating": round(random.uniform(4.4, 4.9), 2),
                "specs": tags,
                "tags": tags,
                "description": f"{name},主打{tags[0]}。",
            }
        )
    return out


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


ORDER_STATUS = ["pending", "paid", "shipped", "delivered", "after_sale", "cancelled"]
COURIERS = ["顺丰", "京东", "中通", "圆通", "韵达"]
USERS = ["U001", "U002", "U003", "U004", "U005"]


def build_orders(products: list[dict]) -> list[dict]:
    out: list[dict] = []
    for i in range(30):
        oid = f"E2026030{(i + 1):05d}"  # E2026030 + 5-digit seq → 13 chars
        prod = random.choice(products)
        qty = random.randint(1, 3)
        status = random.choice(ORDER_STATUS)
        order = {
            "order_id": oid,
            "user_id": random.choice(USERS),
            "product_id": prod["product_id"],
            "product_name": prod["name"],
            "quantity": qty,
            "amount": prod["price"] * qty,
            "status": status,
            "created_at": f"2026-03-{random.randint(1, 28):02d}",
            "courier": random.choice(COURIERS) if status in {"shipped", "delivered"} else None,
            "tracking_no": f"SF{random.randint(100000000, 999999999)}" if status in {"shipped", "delivered"} else None,
            "address": "北京市海淀区中关村大街 1 号",
        }
        out.append(order)
    return out


# ---------------------------------------------------------------------------
# FAQ documents
# ---------------------------------------------------------------------------


FAQ_DOCS = [
    (
        "shipping_time",
        "发货时效",
        ["发货", "几天", "多久", "什么时候发", "时效"],
        """付款后默认 24 小时内发货,偏远地区(西藏、新疆等)48 小时内发货。\n\n大型家电、定制类商品的发货时效以商品详情页标注为准,通常为 3-7 天。""",
    ),
    (
        "shipping_fee",
        "运费政策",
        ["运费", "包邮", "邮费", "运费多少"],
        """全国默认包邮(不含港澳台、海外)。订单金额 < 49 元加收 6 元基础运费;\n大件家电、家具类商品按重量/体积单独计算,详见商品页。""",
    ),
    (
        "return_window",
        "七天无理由",
        ["七天", "无理由", "退货", "退款", "无理由退货"],
        """支持 7 天无理由退货:商品需未拆封或不影响二次销售。生鲜、定制、贴身用品不支持无理由退货。\n\n用户发起退货后,需在 48 小时内寄出商品,客服收到后 3 个工作日内完成审核。""",
    ),
    (
        "refund_timeline",
        "退款到账时间",
        ["退款", "几天到账", "退款时效", "多久退款"],
        """退款审核通过后,退款将原路返回:\n- 微信/支付宝: 1-3 个工作日\n- 银行卡: 3-7 个工作日\n- 信用卡: 7-15 个工作日\n如超时未到账,请联系客服并提供退款单号。""",
    ),
    (
        "warranty",
        "保修政策",
        ["保修", "保修期", "三包", "维修"],
        """所有自营商品默认享受国家三包政策(7 天退、15 天换、1 年保修)。\n\n手机、电脑等数码产品可在商品详情页购买延保服务,最长可至 3 年。""",
    ),
    (
        "invoice",
        "发票开具",
        ["发票", "开票", "增值税", "电子发票"],
        """支持电子普通发票与增值税专用发票:\n1. 下单时选择"开具发票"\n2. 增专需提供公司全称、税号、开户行、地址电话\n电子发票将在订单完成后 48 小时内发送至预留邮箱。""",
    ),
    (
        "membership",
        "会员体系",
        ["会员", "积分", "等级", "成长值"],
        """会员等级: 新人 / 银卡 / 金卡 / 黑卡。每消费 1 元 = 1 成长值。\n金卡享受额外 95 折,黑卡享受额外 9 折并赠送生日礼包。""",
    ),
    (
        "coupon",
        "优惠券使用",
        ["优惠券", "怎么用", "叠加", "满减"],
        """优惠券可在结算页"使用优惠券"中选择。\n\n满减券与会员折扣可叠加,品类券与店铺券不可叠加。每笔订单最多使用 1 张满减券 + 1 张品类券。""",
    ),
    (
        "payment",
        "支付方式",
        ["支付", "支付方式", "怎么付", "分期", "花呗"],
        """支持微信支付、支付宝、银联、Apple Pay,部分商品支持花呗 / 京东白条 3-12 期分期。""",
    ),
    (
        "address_change",
        "修改收货地址",
        ["地址", "改地址", "修改地址", "收货地址"],
        """订单未发货前可在"我的订单 → 修改地址"中自助修改。\n已发货订单需联系快递员或客服协助拦截,无法保证 100% 修改成功。""",
    ),
    (
        "cancel_order",
        "取消订单",
        ["取消订单", "取消", "不要了"],
        """未发货订单可自助取消。已发货订单请拒收或签收后联系客服走退货流程。\n预售商品支付定金后定金不退,但尾款部分支持取消。""",
    ),
    (
        "out_of_stock",
        "缺货补货",
        ["缺货", "补货", "什么时候到货", "没货"],
        """缺货商品页面会显示"到货通知"按钮,点击后我们会在第一时间短信/站内信提醒您。\n\n大部分自营商品的补货周期为 3-7 天。""",
    ),
    (
        "contact_support",
        "联系客服",
        ["联系客服", "人工", "客服电话", "投诉"],
        """在线客服: App 我的页面 → 在线客服(7x24h 智能,9-22 点人工)。\n投诉热线: 400-800-1234。""",
    ),
    (
        "price_protect",
        "价保政策",
        ["价保", "降价", "保价", "差价"],
        """支持下单 30 天内的价保:同型号同卖家降价可申请补差。\n大促期间(618、双 11)价保规则以活动页公告为准。""",
    ),
    (
        "exchange",
        "换货流程",
        ["换货", "换码", "换尺码", "换颜色"],
        """7 天内可申请同款换货(尺码/颜色),换货只支持 1 次。\n仓库收到原商品后 2 个工作日内寄出新商品,运费由我方承担。""",
    ),
]


def build_faq_docs() -> list[tuple[str, str]]:
    files: list[tuple[str, str]] = []
    for topic, title, keywords, body in FAQ_DOCS:
        kw = ", ".join(keywords)
        content = (
            f"---\n"
            f"topic: {topic}\n"
            f"title: {title}\n"
            f"keywords: [{kw}]\n"
            f"---\n\n"
            f"# {title}\n\n{body}\n"
        )
        files.append((f"{topic}.md", content))
    return files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    products = build_products()
    (DATA / "products.json").write_text(
        json.dumps(products, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    orders = build_orders(products)
    (DATA / "orders.json").write_text(
        json.dumps(orders, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for fn, content in build_faq_docs():
        (FAQ / fn).write_text(content, encoding="utf-8")

    print(
        f"Wrote {len(products)} products, {len(orders)} orders, "
        f"{len(FAQ_DOCS)} faq docs into {DATA}"
    )


if __name__ == "__main__":
    main()
