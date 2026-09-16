from typing import Dict
from zhipuai import ZhipuAI
from config import ZHIPU_API_KEY, CHAT_MODEL, MAX_RETRY, TOP_K_RERANK
from agent.state import AgentState
from agent.tools import call_tool

# 初始化客户端
client = ZhipuAI(api_key=ZHIPU_API_KEY)


# ============ 节点1：规划 ============

def plan_node(state: AgentState) -> Dict:
    """
    决定下一步：需要检索吗？检索什么？
    
    - 首次进入：把原问题改写成多个检索 query
    - 再次进入：基于已有资料，判断是否够用，不够则生成新的查询
    """
    question = state["question"]
    documents = state.get("documents", [])
    
    # ----- 首次规划 -----
    if not documents:
        prompt = f"""请把下面的问题改写成3个不同角度的检索查询，每行一个，不要编号：
问题：{question}"""
        
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        text = response.choices[0].message.content
        queries = [q.strip() for q in text.strip().split('\n') if q.strip()][:3]
        
        return {
            "queries": queries,
            "need_retrieve": True,
            "need_retry": False,
        }
    
    # ----- 再次规划：评估已有资料是否够用 -----
    context_preview = "\n---\n".join([d["content"][:200] for d in documents[:3]])
    
    prompt = f"""问题：{question}

已有资料（部分）：
{context_preview}

请严格判断：这些资料是否**完整覆盖**了问题的所有子问题？
- 如果完整覆盖，只输出「够了」
- 如果任何一个子问题没有对应资料，请给出还需要检索的2个查询，每行一个，不要编号"""
    
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    text = response.choices[0].message.content.strip()
    
    if "够了" in text:
        return {
            "queries": [],
            "need_retrieve": False,
            "need_retry": False,
        }
    else:
        new_queries = [q.strip() for q in text.split('\n') if q.strip()][:2]
        return {
            "queries": new_queries,
            "need_retrieve": True,
            "need_retry": True,
        }


# ============ 节点2：检索 ============

def retrieve_node(state: AgentState) -> Dict:
    """
    执行检索：对当前 queries 逐个调用工具，合并结果并去重。
    """
    queries = state["queries"]
    existing = state.get("documents", [])
    
    # 已有内容的集合，用于去重
    existing_contents = {d["content"][:50] for d in existing}
    
    new_documents = []
    for q in queries:
        # 同时用两种工具检索，取并集
        vector_results = call_tool("vector_search", q, top_k=3)
        keyword_results = call_tool("keyword_search", q, top_k=3)
        
        for r in vector_results + keyword_results:
            key = r["content"][:50]
            if key not in existing_contents:
                existing_contents.add(key)
                new_documents.append(r)
    
    return {
        "documents": existing + new_documents,
    }


# ============ 节点3：评估 ============

def evaluate_node(state: AgentState) -> Dict:
    """
    评估检索结果，决定是否需要再检索一轮。
    主要是限制重试次数，避免无限循环。
    """
    retry_count = state.get("retry_count", 0) + 1
    
    # 超过最大重试次数，强制结束检索
    if retry_count >= MAX_RETRY:
        return {
            "need_retry": False,
            "retry_count": retry_count,
        }
    
    # 没有检索到任何内容，也不必重试
    if not state.get("documents"):
        return {
            "need_retry": False,
            "retry_count": retry_count,
        }
    
    return {
        "retry_count": retry_count,
    }


# ============ 节点4：生成 ============

def generate_node(state: AgentState) -> Dict:
    """
    基于检索到的资料生成最终回答。
    """
    question = state["question"]
    documents = state.get("documents", [])
    
    if not documents:
        return {"answer": "知识库中没有找到相关资料，无法回答这个问题。"}
    
    # 拼接上下文
    context = "\n\n---\n\n".join([
        f"【来源：{d['source']}】\n{d['content']}"
        for d in documents[:TOP_K_RERANK]
    ])
    
    system_prompt = (
        "你是一个严格基于资料回答问题的助手。"
        "请只使用下面提供的参考资料来回答问题。"
        "如果参考资料中没有相关信息，必须明确回答'资料中未提及'，"
        "绝对不要使用你自己的先验知识来补充或推断。\n\n"
        f"参考资料：\n{context}"
    )
    
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.3
    )
    
    return {"answer": response.choices[0].message.content}