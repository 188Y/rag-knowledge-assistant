import json
import re
from typing import Dict, List

from zhipuai import ZhipuAI

from config import ZHIPU_API_KEY, CHAT_MODEL, MAX_RETRY, TOP_K_RERANK
from agent.state import AgentState
from agent.tools import call_tool, TOOLS

# 初始化客户端
client = ZhipuAI(api_key=ZHIPU_API_KEY)

# 允许调度的工具名（防止模型幻觉出不存在的工具）
ALLOWED_TOOLS = set(TOOLS.keys())

# 模型输出查询列表时的列表符号 / 提示语
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*•·]|\d+\s*[.、)])\s*")
_PREAMBLE_RE = re.compile(
    r"^(?:还需要|需要|请|补充|以下|资料|这些|上述|不够|无需|查询[:：]?$)"
)


def _format_history_prompt(history: List[dict]) -> str:
    """将多轮对话历史格式化为清晰的对话背景提示"""
    if not history:
        return ""
    lines = ["【前序对话历史】"]
    for msg in history[-6:]:  # 最多取最近6条（3轮）
        role = "用户" if msg.get("role") == "user" else "助手"
        content = msg.get("content", "").strip()
        if content:
            lines.append(f"{role}：{content}")
    return "\n".join(lines) + "\n\n"


def _chat(prompt: str, temperature: float, system: str = "", history: List[dict] = None) -> str:
    """统一的 LLM 调用入口。失败时抛异常，由各节点决定降级策略。"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if history:
        for msg in history[-6:]:
            role = msg.get("role")
            content = msg.get("content", "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return (response.choices[0].message.content or "").strip()


def _append_error(state: AgentState, message: str) -> List[str]:
    """state 里没有 reducer，累加必须显式合并"""
    return list(state.get("tool_errors", [])) + [message]


def _parse_query_lines(text: str, limit: int) -> List[str]:
    """从模型输出里抽取查询行。

    模型经常带上"还需要检索的2个查询："这类提示语或列表符号，直接 split 会把它
    当成一条查询，因此这里做一层清洗。
    """
    queries = []
    for raw in (text or "").split("\n"):
        line = _LIST_PREFIX_RE.sub("", raw).strip()
        if len(line) < 3:
            continue
        if line.endswith(("：", ":")) or _PREAMBLE_RE.match(line):
            continue
        queries.append(line)
        if len(queries) >= limit:
            break
    return queries


def _default_plan(question: str) -> List[dict]:
    """工具规划失败时的兜底：走默认 RAG 双路检索"""
    return [
        {"name": "vector_search", "query": question},
        {"name": "keyword_search", "query": question},
    ]


def _parse_tool_plan(text: str, question: str) -> List[dict]:
    """解析模型输出的工具计划。

    模型返回的 JSON 结构不稳定（顶层是 list、tools 是 dict、混入解释文字等），
    这里全部兜住并返回空列表，绝不抛异常。
    """
    if not text:
        return []

    cleaned = text.replace("```json", "").replace("```", "").strip()
    try:
        plan = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError, TypeError):
        return []

    raw_tools = plan.get("tools", []) if isinstance(plan, dict) else plan
    if isinstance(raw_tools, dict):
        raw_tools = [raw_tools]
    if not isinstance(raw_tools, list):
        return []

    tools = []
    for item in raw_tools:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("tool")
        if name not in ALLOWED_TOOLS:
            continue
        query = item.get("query") or item.get("question") or question
        tools.append({"name": name, "query": str(query)})
        if len(tools) >= 3:
            break
    return tools


def _select_documents(documents: List[dict], limit: int) -> List[dict]:
    """从多轮检索结果里挑出不超过 limit 条，优先保留最新一轮的内容。

    最新一轮是 plan 判定"资料覆盖不全"后专门补检索的，最可能包含缺失信息；
    否则 generate 的 [:TOP_K_RERANK] 会把新补的资料切掉，重试等于白跑。
    """
    if len(documents) <= limit:
        return documents

    by_round = {}
    for index, doc in enumerate(documents):
        by_round.setdefault(doc.get("round", 0), []).append((index, doc))

    picked = []
    for round_no in sorted(by_round, reverse=True):
        for index, doc in by_round[round_no]:
            if len(picked) >= limit:
                break
            picked.append((index, doc))
        if len(picked) >= limit:
            break

    picked.sort(key=lambda pair: pair[0])
    return [doc for _, doc in picked]


# ============ 节点1：规划 ============

def plan_node(state: AgentState) -> Dict:
    """
    决定下一步：需要检索吗？检索什么？
    
    - 首次进入：把原问题改写成多个检索 query（结合前序对话历史）
    - 再次进入：基于已有资料和工具产出，判断是否够用，不够则生成新的查询
    """
    question = state["question"]
    history = state.get("history", [])
    documents = state.get("documents", [])
    sql_result = state.get("sql_result") or {}
    calc_result = state.get("calc_result", "")
    tool_errors = state.get("tool_errors", [])
    round_no = state.get("retry_count", 0)

    # ----- 首次规划（第0轮）：把问题改写成多个检索角度，若有历史记录则结合上下文理解代词/语义 -----
    if round_no == 0 and not documents and not sql_result.get("text") and not calc_result:
        hist_prompt = _format_history_prompt(history)
        prompt = f"""{hist_prompt}请结合上述背景（若有），把下面的当前问题改写成3个不同角度的检索查询，消除代词歧义，每行一个，不要编号：
当前问题：{question}"""

        try:
            text = _chat(prompt, temperature=0.7)
            queries = _parse_query_lines(text, 3)
            errors = list(state.get("tool_errors", []))
        except Exception as e:
            queries = [question]
            errors = _append_error(state, f"查询改写失败：{e}")

        # queries 会作为候选角度传给 tool_plan_node，避免这次改写白做
        return {
            "queries": queries or [question],
            "need_retrieve": True,
            "need_retry": False,
            "tool_errors": errors,
        }
    
    # ----- 再次规划：评估已有资料和工具产出是否够用 -----
    summary_sections = []
    if documents:
        doc_preview = "\n---\n".join(
            [d.get("content", "")[:200] for d in documents[:TOP_K_RERANK]]
        )
        summary_sections.append(f"【已检索到的知识库资料】\n{doc_preview}")
    if sql_result.get("text"):
        summary_sections.append(f"【已获取的SQL查询结果】\n{sql_result['text'][:300]}")
    if calc_result:
        summary_sections.append(f"【已计算结果】\n{calc_result}")
    if tool_errors:
        summary_sections.append(f"【执行异常】\n" + "\n".join(tool_errors[-2:]))

    context_preview = "\n\n".join(summary_sections) if summary_sections else "（暂无有效材料）"
    tried = state.get("queries", [])
    tried_hint = "、".join(tried) if tried else "（无）"

    prompt = f"""问题：{question}

已经尝试过的查询角度：{tried_hint}

已有资料与工具产出：
{context_preview}

请严格判断：这些资料与数据是否**完整覆盖**了问题的所有要求？
- 如果完整覆盖，只输出「够了」
- 如果任何一个部分没有对应资料或数据，请给出还需要检索/查询的2个具体角度（不要与上面尝试过的重复），每行一个，不要编号"""

    try:
        text = _chat(prompt, temperature=0.3)
    except Exception as e:
        # 评估失败就按"资料已够"处理，避免整条链路报错
        return {
            "queries": [],
            "need_retrieve": False,
            "need_retry": False,
            "tool_errors": _append_error(state, f"覆盖度评估失败：{e}"),
        }

    # 用前缀匹配，"资料还不够了" 这类文本里也含 "够了"，不能再做子串包含判断
    head = text.strip().strip("「」\"'“”")
    if head.startswith("够了"):
        return {
            "queries": [],
            "need_retrieve": False,
            "need_retry": False,
        }

    new_queries = _parse_query_lines(text, 2)
    return {
        "queries": new_queries or [question],
        "need_retrieve": True,
        "need_retry": True,
    }

    
def evaluate_node(state: AgentState) -> Dict:
    """
    评估节点：判断是否需要再检索一轮。

    注意：这里必须显式回写 need_retry，否则 graph 里的条件边读到的是
    plan_node 上一轮写入的 False，多轮检索永远不会触发。
    """
    retry_count = state.get("retry_count", 0) + 1

    # 超过最大重试次数，强制结束
    if retry_count >= MAX_RETRY:
        return {
            "need_retry": False,
            "retry_count": retry_count,
            "evaluation": f"已达到最大检索轮次（{MAX_RETRY}），停止重试",
        }

    # 检查本轮工具产出状态：是否有文档、是否有 SQL 结果、是否有报错
    has_docs = bool(state.get("documents"))
    sql_res = state.get("sql_result") or {}
    has_sql = bool(sql_res.get("text") or sql_res.get("rows"))
    has_calc = bool(state.get("calc_result"))
    tool_errors = state.get("tool_errors", [])

    # 如果没有任何有效工具产出，且存在报错，允许进入 plan 再重试调整
    # 如果没有任何有效产出也没有报错，说明确实没有相关内容，停止重试避免死循环
    if not has_docs and not has_sql and not has_calc and not tool_errors:
        return {
            "need_retry": False,
            "retry_count": retry_count,
            "evaluation": "未检索到任何有效信息且无执行错误，停止重试",
        }

    # 交给 plan 再判断资料/数据覆盖度
    return {
        "need_retry": True,
        "retry_count": retry_count,
        "evaluation": "待 plan 复核资料与数据完整度",
    }


def tool_plan_node(state: AgentState) -> Dict:
    """
    工具规划节点：让大模型输出一个工具调用计划。
    """
    question = state["question"]
    history = state.get("history", [])
    queries = state.get("queries", [])
    candidate_hint = "、".join(queries) if queries else question
    retry_count = state.get("retry_count", 0)
    tool_errors = state.get("tool_errors", [])
    sql_result = state.get("sql_result") or {}

    # 如果是重试轮次，注入前序工具执行反馈与错误提示
    feedback_lines = []
    if retry_count > 0:
        feedback_lines.append("【前序工具执行反馈】")
        if tool_errors:
            feedback_lines.append("之前遇到以下异常，请调整工具或查询语句，避免重复相同错误：")
            for err in tool_errors[-3:]:
                feedback_lines.append(f"- {err}")
        if sql_result.get("text"):
            feedback_lines.append(f"之前SQL查询已获得数据，如需对比或补充请针对性调用其它工具：\n{sql_result['text'][:200]}")
        feedback_lines.append("")
    feedback_prompt = "\n".join(feedback_lines)
    hist_prompt = _format_history_prompt(history) if retry_count == 0 else ""

    prompt = f"""{hist_prompt}{feedback_prompt}你是一个工具规划助手。请根据用户问题及前序执行情况，决定调用哪些工具。

可用工具：
- vector_search：语义检索本地文档，适合概念、流程、制度类问题
- keyword_search：关键词精确检索本地文档，适合专有名词、代码
- calculator：数学计算
- sql_query：查询结构化数据库（销售、统计），适合数据类问题

请输出 JSON 格式的工具调用计划，例如：
{{
  "tools": [
    {{"name": "sql_query", "query": "华东区销售额"}},
    {{"name": "vector_search", "query": "华东区策略目标"}}
  ]
}}

要求：
1. 只输出 JSON，不要其他文字
2. 最多选择 3 个工具
3. 每个工具的 query 要清晰、具体
4. 若前序已有异常报错，必须尝试修正参数或换用有效工具

参考的候选查询角度：{candidate_hint}

用户问题：{question}
"""

    try:
        text = _chat(prompt, temperature=0.1)
        errors = list(state.get("tool_errors", []))
    except Exception as e:
        text = ""
        errors = _append_error(state, f"工具规划失败：{e}")

    tools = _parse_tool_plan(text, question)
    if not tools:
        # 解析失败（或模型没给出有效工具），降级为默认 RAG 双路检索
        errors = errors + ["工具计划无效，已降级为默认检索"]
        tools = _default_plan(question)

    return {"tool_plan": tools, "tool_errors": errors}

# ============ 节点2：检索 ============

def retrieve_node(state: AgentState) -> Dict:
    """根据 tool_plan 执行工具调用。

    两点约束：
    1. 每个工具单独 try/except —— 单个工具失败（网络/LLM 超时）不能拖垮整个 Agent；
    2. 新结果必须累加到历史结果上 —— state 没有 reducer，直接返回会把上一轮的
       web/calc/sql 产出覆盖掉，多轮检索时会"失忆"。
    """
    tool_plan = state.get("tool_plan", [])
    existing = state.get("documents", [])
    existing_contents = {d.get("content", "")[:50] for d in existing}

    round_no = state.get("retry_count", 0)
    calc_results = [c for c in [state.get("calc_result", "")] if c]
    sql_result = state.get("sql_result") or {}
    sql_texts = [sql_result["text"]] if sql_result.get("text") else []
    sql_statements = [s for s in [state.get("sql_query", "")] if s]
    tool_errors = list(state.get("tool_errors", []))
    new_documents = []

    for tool in tool_plan:
        if not isinstance(tool, dict):
            tool_errors.append(f"忽略非法的工具调用项：{tool!r}")
            continue

        name = tool.get("name", "")
        query = tool.get("query") or tool.get("question") or state["question"]

        if name not in ALLOWED_TOOLS:
            tool_errors.append(f"模型请求了不存在的工具：{name}")
            continue

        try:
            if name in ("vector_search", "keyword_search"):
                extra_kwargs = {"queries": state.get("queries", [])} if name == "vector_search" else {}
                for r in call_tool(name, query=query, top_k=3, **extra_kwargs) or []:
                    content = r.get("content", "")
                    key = content[:50]
                    if key and key not in existing_contents:
                        existing_contents.add(key)
                        new_documents.append({
                            "content": content,
                            "source": r.get("source", "unknown"),
                            "round": round_no,
                        })

            elif name == "calculator":
                content = (call_tool("calculator", query=query) or {}).get("content", "")
                if content and content not in calc_results:
                    calc_results.append(content)

            elif name == "sql_query":
                r = call_tool("sql_query", query=query) or {}
                if r.get("text"):
                    sql_texts.append(r["text"])
                if r.get("sql"):
                    sql_statements.append(r["sql"])
                if r.get("success"):
                    # 只用成功的结果覆盖 rows/columns；失败的详情已在 text 里保留，
                    # 否则计划里后面的查询失败会抹掉前面已查到的数据
                    sql_result = r

        except Exception as e:
            # 单工具失败只记录，继续执行其它工具
            tool_errors.append(f"{name} 执行失败：{e}")

    if sql_texts:
        merged = dict(sql_result)
        # 多轮/多次查询的文本按出现顺序去重拼接，行/列取最近一次
        merged["text"] = "\n\n".join(dict.fromkeys(sql_texts))
        sql_result = merged

    return {
        "documents": existing + new_documents,
        "calc_result": "\n".join(dict.fromkeys(calc_results)),
        "sql_query": "\n".join(dict.fromkeys(sql_statements)),
        "sql_result": sql_result,
        "tool_errors": tool_errors,
    }



# ============ 节点4：生成 ============

def generate_node(state: AgentState) -> Dict:
    """
    基于知识库资料和各工具的产出生成最终回答。
    """
    question = state["question"]
    history = state.get("history", [])
    documents = state.get("documents", [])
    calc_result = state.get("calc_result", "")
    sql_result = state.get("sql_result") or {}
    
    # 各来源的产出拼成材料块，只要任一路有内容就不算"没找到"
    sections = []
    
    if documents:
        # 多轮检索时优先保留最新一轮补检索到的资料
        selected = _select_documents(documents, TOP_K_RERANK)
        context = "\n\n---\n\n".join([
            f"【来源：{d.get('source', 'unknown')}】\n{d.get('content', '')}"
            for d in selected
        ])
        sections.append(f"【知识库资料】\n{context}")
    
    if calc_result:
        sections.append(f"【计算器结果】\n{calc_result}")
    
    if sql_result.get("text"):
        sections.append(f"【SQL 查询结果】\n{sql_result['text']}")
    
    tool_errors = state.get("tool_errors", [])
    if tool_errors:
        # 让模型知道哪个工具失败了，避免把"工具失败"说成"材料中未提及"
        errors_text = "\n".join(dict.fromkeys(tool_errors))
        sections.append(f"【工具调用异常】\n{errors_text}")
    
    if not sections:
        return {"answer": "知识库中没有找到相关资料，无法回答这个问题。"}
    
    system_prompt = (
        "你是一个严格基于给定材料回答问题的助手。"
        "请只使用下面提供的材料来回答问题。"
        "如果材料中没有相关信息，必须明确回答'材料中未提及'，"
        "绝对不要使用你自己的先验知识来补充或推断。"
        "其中计算器结果和 SQL 查询结果是工具直接产出的，"
        "属于权威结果，请直接采用，不需要再找资料佐证。"
        "如果材料中出现【工具调用异常】，说明对应工具本次调用失败，"
        "请如实说明该部分信息未能获取，不要编造。\n\n"
        + "\n\n".join(sections)
    )
    
    try:
        answer = _chat(question, temperature=0.3, system=system_prompt, history=history)
    except Exception as e:
        # LLM 失败时给出兜底回答，避免整个接口 500
        answer = f"生成回答失败：{e}"

    return {"answer": answer}