import os
import random
import sqlite3
from datetime import date, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "sales.db")

# 固定随机种子：同一台机器上重复初始化得到同样的数据，便于复现问题
random.seed(42)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            region TEXT NOT NULL,
            product TEXT NOT NULL,
            amount REAL NOT NULL,
            quantity INTEGER NOT NULL,
            sale_date TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS strategy (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            region TEXT NOT NULL,
            year INTEGER NOT NULL,
            target_amount REAL NOT NULL,
            focus_product TEXT
        )
    """)

    cursor.execute("DELETE FROM sales")
    cursor.execute("DELETE FROM strategy")

    # ---- 种子数据：以"今天"为锚动态生成，任何时间初始化都覆盖最近 3 个月 ----
    # 原来写死 2026-07 ~ 2026-09 的静态日期，随时间推移"本月/上月"类问题必然查空。
    # 现在日期 = 今天 - 回溯天数（0/3/10/20/35/50/70/88 天），
    # 保证当月、上月、上上月都必有数据，且永远不过期。
    today = date.today()
    lookback_days = [0, 3, 10, 20, 35, 50, 70, 88]
    products = ["产品A", "产品B", "产品C"]

    sales_rows = []
    for region, target in (("华东", 300000), ("华北", 200000), ("华南", 250000)):
        for i, back in enumerate(lookback_days):
            amount = random.randint(55000, 145000)
            # 数量与金额大致成比例（约 380~420 元/件），保证数据自洽
            quantity = round(amount / random.uniform(380, 420))
            sales_rows.append((
                region,
                products[(i + len(region)) % len(products)],
                amount,
                quantity,
                (today - timedelta(days=back)).isoformat(),
            ))

    cursor.executemany(
        "INSERT INTO sales (region, product, amount, quantity, sale_date) VALUES (?, ?, ?, ?, ?)",
        sales_rows
    )

    # 年度策略同样跟随当前年份，避免"今年目标"查不到
    cursor.executemany(
        "INSERT INTO strategy (region, year, target_amount, focus_product) VALUES (?, ?, ?, ?)",
        [
            ("华东", today.year, 300000, "产品A"),
            ("华北", today.year, 200000, "产品A"),
            ("华南", today.year, 250000, "产品B"),
        ]
    )
    conn.commit()
    conn.close()

    dates = [row[4] for row in sales_rows]
    print(f"数据库已初始化：{DB_PATH}")
    print(f"共 {len(sales_rows)} 行销售记录，日期范围 {min(dates)} ~ {max(dates)}（以今天 {today} 为锚）")


if __name__ == "__main__":
    init_db()
