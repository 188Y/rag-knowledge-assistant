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
            "sql_result": dict,        # SQL 工具的产出（含 sql / text / rows）
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
        "sql_query": "",
        "sql_result": {},
        "tool_errors": [],
    }
    
    result = agent_app.invoke(initial_state)
    
    return {
        "answer": result.get("answer", ""),
        "documents": result.get("documents", []),
        "retry_count": result.get("retry_count", 0),
        "sql_result": result.get("sql_result", {}),
        "calc_result": result.get("calc_result", ""),
        "tool_errors": result.get("tool_errors", []),
    }