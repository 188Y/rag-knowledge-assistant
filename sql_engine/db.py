import os
import sqlite3
import threading
from contextlib import closing

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "sales.db"
)

# 单次查询最多取回的原始行数，避免 SELECT * 把整库读进内存再塞进上下文
MAX_ROWS = 200

# 线程局部的只读连接缓存：Flask 默认 threaded=True，每个工作线程
# 复用同一个只读连接，避免每次查询都重新 open（SQLite 打开文件并不便宜）
_thread_local = threading.local()


def _get_ro_connection():
    """获取当前线程专用的只读连接（首次调用时创建并缓存）。

    线程销毁时连接对象随线程局部存储一起被 GC 关闭，无需显式清理。
    """
    conn = getattr(_thread_local, "ro_conn", None)
    if conn is None:
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(
                f"数据库文件不存在：{DB_PATH}，请先执行 python data/init_db.py 初始化"
            )
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        _thread_local.ro_conn = conn
    return conn


def get_connection(read_only: bool = True):
    """获取 SQLite 连接。

    read_only=True 时用只读 URI 打开：即使上层 SQL 校验被绕过，
    写操作也会被 SQLite 直接拒绝（attempt to write a readonly database）。
    read_only 模式返回线程局部复用连接（调用方不要 close 它）。
    """
    if read_only:
        return _get_ro_connection()

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"数据库文件不存在：{DB_PATH}，请先执行 python data/init_db.py 初始化"
        )
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_sales_date_range():
    """查询 sales 表的日期范围（MIN/MAX），供 Text2SQL prompt 动态注入。

    用一次性连接查询（只在进程启动后调用一次并被调用方缓存）。
    库文件不存在或表为空时返回 None。
    """
    if not os.path.exists(DB_PATH):
        return None
    try:
        with closing(sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)) as conn:
            row = conn.execute(
                "SELECT MIN(sale_date), MAX(sale_date) FROM sales"
            ).fetchone()
    except sqlite3.Error:
        return None
    if not row or not row[0]:
        return None
    return row[0], row[1]


def execute_sql(sql: str, max_rows: int = MAX_ROWS) -> dict:
    """执行只读 SQL 并返回结果。

    所有异常都转成 {"success": False, "error": ...}，不向上抛；
    连接为线程局部复用连接，不在此处关闭。
    """
    try:
        conn = _get_ro_connection()
        cursor = conn.cursor()
        cursor.execute(sql)

        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        if not columns:
            # 非查询语句不会产生结果集
            return {
                "success": True,
                "columns": [],
                "rows": [],
                "error": None,
                "truncated": False,
            }

        # 多取一行用于判断是否被截断
        fetched = cursor.fetchmany(max_rows + 1)
        truncated = len(fetched) > max_rows
        rows = [dict(zip(columns, row)) for row in fetched[:max_rows]]

        return {
            "success": True,
            "columns": columns,
            "rows": rows,
            "error": None,
            "truncated": truncated,
        }
    except Exception as e:
        return {
            "success": False,
            "columns": [],
            "rows": [],
            "error": str(e),
            "truncated": False,
        }
