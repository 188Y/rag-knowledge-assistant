from typing import List, Dict
from rag_engine import search_enhanced, search_bm25


def tool_vector_search(query: str, top_k: int = 3) -> List[Dict]:
    """
    向量检索工具：基于语义相似度检索文档片段。
    适合：概念性、描述性问题。
    """
    results = search_enhanced(query, top_k=top_k)
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


# 工具注册表：名称 -> 函数
TOOLS = {
    "vector_search": tool_vector_search,
    "keyword_search": tool_keyword_search,
}


def call_tool(name: str, query: str, top_k: int = 3) -> List[Dict]:
    """统一调用入口"""
    if name not in TOOLS:
        raise ValueError(f"未知工具：{name}")
    return TOOLS[name](query, top_k=top_k)