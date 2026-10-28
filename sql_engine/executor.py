import re

from sql_engine.text_to_sql import natural_language_to_sql
from sql_engine.db import execute_sql

# ============ SQL 安全校验 ============
# 只校验"是否是安全的单条只读查询"，校验用的是剥掉注释后的文本，
# 真正执行仍用模型原始输出（SQLite 自己能正确处理行内注释）。

_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_STRING_LITERAL_RE = re.compile(r"'(?:[^']|'')*'")
_ALLOWED_START_RE = re.compile(r"^\s*(select|with)\b", re.I)

# 写操作关键字：CTE 里也可能藏写操作（WITH t AS (DELETE FROM ...) SELECT ...），
# 所以不能只看首关键字，还要全句扫描；
# load_extension 属于纵深防御补强——Python sqlite3 默认禁用扩展加载，
# 但显式拦截更稳妥
_WRITE_KEYWORD_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|"
    r"pragma|vacuum|reindex|truncate|grant|revoke|begin|savepoint|"
    r"load_extension)\b",
    re.I,
)


def normalize_sql(sql: str) -> str:
    """去掉注释并压缩空白，仅用于校验，不作为执行语句"""
    if not sql:
        return ""
    without_block = _BLOCK_COMMENT_RE.sub(" ", sql)
    without_line = _LINE_COMMENT_RE.sub(" ", without_block)
    return re.sub(r"\s+", " ", without_line).strip()


def validate_select(sql: str) -> tuple:
    """校验 SQL 是否为安全的单条只读查询。

    返回 (是否合法, 原因)。合法的形式包括：
    - SELECT ...（含小写、带前导注释、带结尾分号）
    - WITH ... AS (...) SELECT ...（CTE 聚合，Text2SQL 常用）
    """
    if not sql or not sql.strip():
        return False, "模型未生成有效的 SQL"

    normalized = normalize_sql(sql)
    if not normalized:
        # 例如 -- ERROR: 无法生成
        return False, "模型未生成有效的 SQL"

    if not _ALLOWED_START_RE.match(normalized):
        return False, "只允许 SELECT / WITH 查询"

    # 去掉字符串字面量再判断，避免把 'delete' 这类普通内容误判成写操作
    without_literals = _STRING_LITERAL_RE.sub("''", normalized)

    if _WRITE_KEYWORD_RE.search(without_literals):
        return False, "查询中包含写操作关键字，已拒绝执行"

    body = without_literals.rstrip().rstrip(";")
    if ";" in body:
        return False, "只允许单条 SQL 语句"

    return True, ""


def query_database(question: str) -> dict:
    try:
        sql = natural_language_to_sql(question)
    except Exception as e:
        return {
            "success": False,
            "sql": "",
            "columns": [],
            "rows": [],
            "error": f"SQL 生成失败：{e}",
        }

    is_valid, reason = validate_select(sql)
    if not is_valid:
        return {
            "success": False,
            "sql": sql,
            "columns": [],
            "rows": [],
            "error": reason,
        }

    result = execute_sql(sql)
    result["sql"] = sql
    return result


def format_result_for_llm(result: dict) -> str:
    """把查询结果格式化成给大模型看的文本，始终带上 SQL 原文便于排查与自我纠正"""
    sql = (result.get("sql") or "").strip()
    columns = result.get("columns") or []
    rows = result.get("rows") or []

    lines = []
    if sql:
        lines.append(f"SQL: {sql}")

    if not result.get("success"):
        lines.append(f"查询失败：{result.get('error') or '未知错误'}")
        return "\n".join(lines)

    if not rows:
        lines.append("查询结果为空。")
        return "\n".join(lines)

    lines.append(" | ".join(columns))
    lines.append("-" * 40)
    for row in rows[:20]:
        lines.append(" | ".join(str(row.get(col, "")) for col in columns))

    if len(rows) > 20:
        lines.append(f"...（本页共 {len(rows)} 行，仅展示前 20 行）")
    if result.get("truncated"):
        lines.append("（结果行数过多，已截断）")

    return "\n".join(lines)