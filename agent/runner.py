from agent.graph import agent_app
from agent.state import AgentState


def run_agent(question: str) -> dict:
    """
    运行 Agent，返回最终结果。
    
    返回：
        {
            "answer": str,
            "documents": List[dict],   # 所有检索到的文档
            "retry_count": int,
        }
    """
    initial_state: AgentState = {
        "question": question,
        "queries": [],
        "need_retrieve": False,
        "need_retry": False,
        "documents": [],
        "retry_count": 0,
        "evaluation": "",
        "answer": "",
    }
    
    result = agent_app.invoke(initial_state)
    
    return {
        "answer": result["answer"],
        "documents": result.get("documents", []),
        "retry_count": result.get("retry_count", 0),
    }