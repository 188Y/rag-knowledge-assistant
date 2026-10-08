from typing import List, Dict
from rag_engine import search_enhanced, search_bm25
from sql_engine import query_database, format_result_for_llm


try:
    from sympy import sympify
    from sympy.core.sympify import SympifyError
    CALC_AVAILABLE = True
except ImportError:
    CALC_AVAILABLE = False


def tool_vector_search(query: str, top_k: int = 3, queries: List[str] = None) -> List[Dict]:
    """
    向量检索工具：基于语义相似度检索文档片段。
    适合：概念性、描述性问题。
    queries：已改写好的多角度查询（来自 plan_node），传入则不再重复调用 LLM 改写。
    """
    results = search_enhanced(query, top_k=top_k, queries=queries)
    return [
        {"content": r["content"], "source": r.get("source", "unknown")}
        for r in results
    ]


def tool_keyword_search(query: str, top_k: int = 3) -> List[Dict]:
    """
    关键词检索工具：基于 BM25 精确匹配检索文档片段。
    适合：专有名词、代码、术语。
    """
    results = search_bm25(query, top_k=top_k)
    return [
        {"content": r["content"], "source": r.get("source", "unknown")}
        for r in results
    ]

# ============ 工具3：计算器 ============


def tool_calculator(expression: str) -> Dict:
    if not CALC_AVAILABLE:
        return {"content": "计算器不可用", "source": "calculator"}
    try:
        expr = sympify(expression)
        result = expr.evalf() if expr.is_real else expr
        try:
            result_str = str(int(result)) if result == int(result) else str(round(float(result), 6))
        except (TypeError, ValueError):
            result_str = str(result)
        return {"content": f"{expression} = {result_str}", "source": "calculator"}
    except SympifyError:
        return {"content": f"无法解析：{expression}", "source": "calculator"}
    except Exception as e:
        return {"content": f"计算失败：{str(e)}", "source": "calculator"}


def tool_sql_query(question: str) -> Dict:
    """Text2SQL 工具。任何异常都转成失败结果，避免拖垮整个 Agent。"""
    try:
        result = query_database(question)
    except Exception as e:
        result = {
            "success": False,
            "sql": "",
            "columns": [],
            "rows": [],
            "error": f"SQL 查询异常：{e}",
        }
    result["text"] = format_result_for_llm(result)
    return result


# 工具注册表：名称 -> 函数
# 注意：曾支持 web_search（DuckDuckGo），但当前网络下 duckduckgo.com 不可达，
# 智谱官方联网检索又需要单独资源包，取不到数据反而会误导回答，故整体移除。
TOOLS = {
    "vector_search": tool_vector_search,
    "keyword_search": tool_keyword_search,
    "calculator": tool_calculator,
    "sql_query": tool_sql_query,
}


def call_tool(name, query=None, top_k=3, **kwargs):
    if name not in TOOLS:
        raise ValueError(f"未知工具：{name}")
    if name == "vector_search":
        return TOOLS[name](query, top_k=top_k, queries=kwargs.get("queries"))
    if name == "keyword_search":
        return TOOLS[name](query, top_k=top_k)
    if name == "calculator":
        return TOOLS[name](query or kwargs.get("expression", ""))
    if name == "sql_query":
        return TOOLS[name](query or kwargs.get("question", ""))
