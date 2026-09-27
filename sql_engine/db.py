import os
import sqlite3
from contextlib import closing

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "sales.db"
)

# 单次查询最多取回的原始行数，避免 SELECT * 把整库读进内存再塞进上下文
MAX_ROWS = 200


def get_connection(read_only: bool = True):
    """获取 SQLite 连接。

    read_only=True 时用只读 URI 打开：即使上层 SQL 校验被绕过，
    写操作也会被 SQLite 直接拒绝（attempt to write a readonly database）。
    """
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"数据库文件不存在：{DB_PATH}，请先执行 python data/init_db.py 初始化"
        )

    if read_only:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def execute_sql(sql: str, max_rows: int = MAX_ROWS) -> dict:
    """执行只读 SQL 并返回结果。

    所有异常都转成 {"success": False, "error": ...}，不向上抛；
    连接用 closing() 保证异常路径也会关闭。
    """
    try:
        with closing(get_connection(read_only=True)) as conn:
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