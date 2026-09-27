import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "sales.db")


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

    cursor.executemany(
        "INSERT INTO sales (region, product, amount, quantity, sale_date) VALUES (?, ?, ?, ?, ?)",
        [
            ("华东", "产品A", 120000, 300, "2026-07-15"),
            ("华东", "产品B", 95000, 250, "2026-08-10"),
            ("华东", "产品A", 135000, 320, "2026-08-20"),
            ("华北", "产品A", 80000, 200, "2026-07-20"),
            ("华北", "产品C", 60000, 180, "2026-08-05"),
            ("华南", "产品B", 110000, 280, "2026-07-25"),
            ("华南", "产品A", 100000, 260, "2026-08-15"),
            ("华东", "产品C", 70000, 190, "2026-09-01"),
        ]
    )
    cursor.executemany(
        "INSERT INTO strategy (region, year, target_amount, focus_product) VALUES (?, ?, ?, ?)",
        [
            ("华东", 2026, 300000, "产品A"),
            ("华北", 2026, 200000, "产品A"),
            ("华南", 2026, 250000, "产品B"),
        ]
    )
    conn.commit()
    conn.close()
    print(f"数据库已初始化：{DB_PATH}")


if __name__ == "__main__":
    init_db()