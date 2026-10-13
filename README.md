# RAG 知识库助手

**简体中文** | [English](README_EN.md)

一个把 **RAG 检索**、**LangGraph Agent** 和 **Text2SQL** 拼在一起的问答 Demo：上传自己的文档当知识库，问它问题时它会先检索资料再回答，答不出就说"资料中未提及"；需要数字时还能自己查数据库、做计算。

## 功能

- **知识库问答**：上传 `.txt` / `.pdf` / `.docx`，自动切分入库（ChromaDB）。回答严格基于检索到的片段，不靠模型先验知识编造，缺少依据时明确回答"资料中未提及"。
- **混合检索**：LLM 多角度改写查询 → 向量检索（bge-small-zh）＋ BM25 关键词检索（jieba 分词）→ RRF 融合 → bge-reranker 精排取 Top-3。
- **Agent 模式**：LangGraph 状态图 `plan → tool_plan → retrieve → evaluate → generate`，评估不通过可回到 `plan` 重试（默认最多 2 次）。可自主调用四个工具：向量检索、关键词检索、计算器（sympy）、Text2SQL。
- **Text2SQL**：自然语言直接查 SQLite，内置 `sales`（销售记录）与 `strategy`（年度策略）两张示例表，支持按地区关联做"实际 vs 目标"对比。
- **文档管理**：前端可查看知识库现有文档、按文件名删除。
- **多轮对话**：携带最近 6 条历史消息作为上下文。

## 技术栈

| 层次 | 选型 |
| --- | --- |
| Web 框架 | Flask |
| 大模型 | 智谱 GLM-4-Flash（`zhipuai` SDK） |
| 向量库 | ChromaDB |
| 向量 / 重排模型 | `BAAI/bge-small-zh-v1.5`、`BAAI/bge-reranker-base` |
| 关键词检索 | `rank_bm25` + `jieba` |
| 编排 | LangGraph |
| 计算 / 数据库 | sympy、SQLite |

## 快速开始

```bash
# 1. 安装依赖
pip install flask python-dotenv zhipuai PyPDF2 python-docx chromadb \
  langchain-text-splitters sentence-transformers jieba rank-bm25 langgraph sympy

# 2. 配置智谱 API Key
echo "ZHIPU_API_KEY=你的key" > .env

# 3. 生成示例数据库（Text2SQL 需要）
python data/init_db.py

# 4. 启动
python app.py          # 访问 http://127.0.0.1:5000
```

首次启动会下载两个 BGE 模型（约数百 MB）。国内网络较慢时先执行：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/upload_knowledge` | 上传文档入库（form-data，字段 `file`） |
| `POST` | `/ask` | 普通 RAG 问答 |
| `POST` | `/ask_agent` | Agent 问答，额外返回 `sql_result` / `calc_result` / `retry_count` |
| `GET` | `/documents` | 列出知识库中的文档 |
| `POST` | `/delete_document` | 按文件名删除文档（JSON，字段 `filename`） |

## 配置

模型与检索参数集中在 `config.py`：`CHAT_MODEL`、`CHUNK_SIZE` / `CHUNK_OVERLAP`、`TOP_K_VECTOR` / `TOP_K_BM25` / `TOP_K_RERANK`、`MAX_RETRY`、`CHROMA_PATH`。

## 目录结构

```
my-summarizer/
├── app.py                 # Flask 入口与路由
├── config.py              # 模型 / 检索参数
├── rag_engine.py          # 切分、向量检索、BM25、RRF 融合、重排
├── agent/                 # LangGraph Agent
│   ├── graph.py           #   状态图装配
│   ├── nodes.py           #   plan / tool_plan / retrieve / evaluate / generate
│   ├── tools.py           #   向量检索 / 关键词检索 / 计算器 / Text2SQL
│   ├── state.py           #   AgentState 定义
│   └── runner.py          #   run_agent 入口
├── sql_engine/            # Text2SQL
│   ├── schema.py          #   表结构描述（喂给 LLM）
│   ├── text_to_sql.py     #   自然语言 → SQL
│   ├── executor.py        #   执行与结果格式化
│   └── db.py              #   SQLite 连接
├── data/init_db.py        # 生成示例 sales.db
├── templates/index.html   # 前端单页
└── static/                # main.js / style.css
```

## 许可

MIT License © 2026 188Y — 详见 [LICENSE](LICENSE)。
