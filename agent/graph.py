from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import plan_node, retrieve_node, generate_node, tool_plan_node, evaluate_node

def build_graph():
    """构建 Agent 状态图"""
    
    graph = StateGraph(AgentState)
    
    # 注册节点
    graph.add_node("plan", plan_node)
    graph.add_node("tool_plan", tool_plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("generate", generate_node)
    
    # 入口
    graph.set_entry_point("plan")
    
    # plan → 根据 need_retrieve 决定先做工具规划，还是直接生成
    # 只挂条件边：若再给 plan 加一条无条件边指向 tool_plan，两者会在同一超步
    # 并发执行，retrieve 只能读到上一轮的旧 tool_plan（首轮就是空列表）
    graph.add_conditional_edges(
        "plan",
        lambda s: "tool_plan" if s["need_retrieve"] else "generate",
        {
            "tool_plan": "tool_plan",
            "generate": "generate",
        }
    )
    
    # tool_plan → retrieve
    graph.add_edge("tool_plan", "retrieve")
    
    # retrieve → evaluate
    graph.add_edge("retrieve", "evaluate")
    
    # evaluate → 根据 need_retry 决定回到 plan 还是生成
    graph.add_conditional_edges(
        "evaluate",
        lambda s: "plan" if s["need_retry"] else "generate",
        {
            "plan": "plan",
            "generate": "generate",
        }
    )
    
    # generate → 结束
    graph.add_edge("generate", END)
    
    return graph.compile()


# 全局单例
agent_app = build_graph()