from typing import List, Optional
from agent.graph import agent_app
from agent.state import AgentState


def run_agent(question: str, history: Optional[List[dict]] = None) -> dict:
    """
    运行 Agent，返回最终结果。

    返回：
        {
            "answer": str,
            "documents": List[dict],   # 所有检索到的文档
            "retry_count": int,
            "sql_blocks": List[dict],  # 每次成功 SQL 查询一个块（含 sql / columns / rows / truncated / text）
            "calc_result": str,
            "tool_errors": List[str],
        }
    """
    initial_state: AgentState = {
        "question": question,
        "history": history or [],
        "queries": [],
        "need_retrieve": False,
        "need_retry": False,
        "documents": [],
        "retry_count": 0,
        "evaluation": "",
        "answer": "",
        "tool_plan": [],
        "calc_result": "",
        "sql_blocks": [],
        "tool_errors": [],
    }

    result = agent_app.invoke(initial_state)

    return {
        "answer": result.get("answer", ""),
        "documents": result.get("documents", []),
        "retry_count": result.get("retry_count", 0),
        "sql_blocks": result.get("sql_blocks", []),
        "calc_result": result.get("calc_result", ""),
        "tool_errors": result.get("tool_errors", []),
    }
