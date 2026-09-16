from typing import TypedDict, List

#LangGraph 要求状态是字典结构 所以使用TypedDict
class AgentState(TypedDict):
    """Agent 在整个流程中携带的状态"""
    
    # 输入
    question: str                 # 用户原始问题
    
    # 规划阶段
    queries: List[str]            # 当前要执行的检索查询
    need_retrieve: bool           # 是否需要检索
    need_retry: bool              # 是否需要再检索一轮
    
    # 检索阶段
    documents: List[dict]         # 已检索到的文档片段
                                  # 每项格式：{"content": str, "source": str}
    
    # 评估阶段
    retry_count: int              # 已重试次数
    evaluation: str               # 评估结论（"够了" / "不够"）
    
    # 输出
    answer: str                   # 最终回答