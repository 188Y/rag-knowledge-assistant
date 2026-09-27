import datetime
import re

from zhipuai import ZhipuAI

from config import ZHIPU_API_KEY, CHAT_MODEL
from sql_engine.schema import SCHEMA_DESCRIPTION

client = ZhipuAI(api_key=ZHIPU_API_KEY)

# markdown 围栏（```sql / ```SQL / ```）与 "SQL:" 之类的前缀
_FENCE_RE = re.compile(r"```[a-zA-Z]*")
_LABEL_RE = re.compile(r"^\s*(?:sql|查询语句|语句|result)\s*[:：]\s*", re.I)
# 模型常把问题/解释一起回显，这里定位第一条语句（允许行首带 "SQL:" 标签）
_STATEMENT_RE = re.compile(
    r"(?:^|\n)\s*(?:sql\s*[:：]\s*)?((?:select|with)\b.*)", re.I | re.S
)


def clean_sql_output(text: str) -> str:
    """清掉模型输出里的 markdown 围栏、'SQL:' 前缀等说明文字"""
    if not text:
        return ""

    cleaned = _FENCE_RE.sub("", text).replace("```", "").strip()
    while True:
        stripped = _LABEL_RE.sub("", cleaned).strip()
        if stripped == cleaned:
            break
        cleaned = stripped

    # 模型可能连着问题和解释一起返回（如"问题：xxx\nSQL: SELECT ..."），从第一条语句开始截取
    match = _STATEMENT_RE.search(cleaned)
    if match:
        cleaned = match.group(1).strip()
    return cleaned


def natural_language_to_sql(question: str) -> str:
    """把自然语言问题转成 SQL。失败时返回 '-- ERROR: ...' 而不是抛异常。"""
    today = datetime.date.today().isoformat()

    prompt = f"""你是 SQL 生成助手。根据下面的数据库结构，把用户问题转成 SQLite 的 SELECT 语句。

{SCHEMA_DESCRIPTION}

参考示例：
问题：华东区一共卖了多少钱？
SQL: SELECT SUM(amount) AS total FROM sales WHERE region = '华东'

问题：2026年各地区的销售完成率是多少？
SQL: WITH s AS (SELECT region, SUM(amount) AS amount FROM sales GROUP BY region) SELECT s.region, s.amount, st.target_amount, ROUND(s.amount / st.target_amount, 2) AS rate FROM s JOIN strategy st ON s.region = st.region WHERE st.year = 2026

要求：
1. 只输出一条 SQL，不要解释，不要 Markdown 代码块
2. 只生成只读查询（SELECT，或 WITH ... SELECT），禁止 INSERT/UPDATE/DELETE/DROP
3. 表名、列名必须来自上面的结构，不要臆造；实在无法生成时只输出 -- ERROR: 无法生成
4. 今天是 {today}，涉及"本月、上月、最近"等时间条件时以该日期为基准

问题：{question}
"""

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
    except Exception as e:
        return f"-- ERROR: 调用模型失败：{e}"

    content = response.choices[0].message.content or ""
    sql = clean_sql_output(content)
    return sql or "-- ERROR: 模型未返回内容"