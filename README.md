# RAG 知识库助手

一个把 **RAG 检索**、**LangGraph Agent** 和 **Text2SQL** 拼在一起的问答 Demo：上传自己的文档当知识库，问它问题时它会先检索资料再回答，答不出就说"资料中未提及"；需要数字时还能自己查数据库、做计算。

A Q&A demo combining **RAG retrieval**, a **LangGraph agent**, and **Text2SQL**: upload your own documents as a knowledge base, and it answers strictly from retrieved material — falling back to "not mentioned in the documents" when the sources don't cover it. When numbers are involved, it queries a SQLite database and does the math itself.

---

## 中文

### 功能

- **知识库问答**：上传 `.txt` / `.pdf` / `.docx`，自动切分入库（ChromaDB）。回答严格基于检索到的片段，不靠模型先验知识编造，缺少依据时明确回答"资料中未提及"。
- **混合检索**：LLM 多角度改写查询 → 向量检索（bge-small-zh）＋ BM25 关键词检索（jieba 分词）→ RRF 融合 → bge-reranker 精排取 Top-3。
- **Agent 模式**：LangGraph 状态图 `plan → tool_plan → retrieve → evaluate → generate`，评估不通过可回到 `plan` 重试（默认最多 2 次）。可自主调用四个工具：向量检索、关键词检索、计算器（sympy）、Text2SQL。
- **Text2SQL**：自然语言直接查 SQLite，内置 `sales`（销售记录）与 `strategy`（年度策略）两张示例表，支持按地区关联做"实际 vs 目标"对比。
- **文档管理**：前端可查看知识库现有文档、按文件名删除。
- **多轮对话**：携带最近 6 条历史消息作为上下文。

### 技术栈

| 层次 | 选型 |
| --- | --- |
| Web 框架 | Flask |
| 大模型 | 智谱 GLM-4-Flash（`zhipuai` SDK） |
| 向量库 | ChromaDB |
| 向量 / 重排模型 | `BAAI/bge-small-zh-v1.5`、`BAAI/bge-reranker-base` |
| 关键词检索 | `rank_bm25` + `jieba` |
| 编排 | LangGraph |
| 计算 / 数据库 | sympy、SQLite |

### 快速开始

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

### 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/upload_knowledge` | 上传文档入库（form-data，字段 `file`） |
| `POST` | `/ask` | 普通 RAG 问答 |
| `POST` | `/ask_agent` | Agent 问答，额外返回 `sql_result` / `calc_result` / `retry_count` |
| `GET` | `/documents` | 列出知识库中的文档 |
| `POST` | `/delete_document` | 按文件名删除文档（JSON，字段 `filename`） |

### 配置

模型与检索参数集中在 `config.py`：`CHAT_MODEL`、`CHUNK_SIZE` / `CHUNK_OVERLAP`、`TOP_K_VECTOR` / `TOP_K_BM25` / `TOP_K_RERANK`、`MAX_RETRY`、`CHROMA_PATH`。

### 目录结构

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

---

## English

### Features

- **Knowledge-base Q&A** — Upload `.txt` / `.pdf` / `.docx`; the text is chunked and stored in ChromaDB. Answers are grounded strictly in retrieved chunks, never in the model's prior knowledge, and the assistant explicitly replies "not mentioned in the documents" when the sources fall short.
- **Hybrid retrieval** — LLM query rewriting → vector search (bge-small-zh) plus BM25 keyword search (jieba tokenizer) → RRF fusion → bge-reranker re-ranking, keeping the top 3 chunks.
- **Agent mode** — A LangGraph state machine `plan → tool_plan → retrieve → evaluate → generate`, where a failed evaluation loops back to `plan` (2 retries by default). Four tools are available: vector search, keyword search, a sympy calculator, and Text2SQL.
- **Text2SQL** — Ask questions in plain language against SQLite. Two sample tables ship with the repo: `sales` and `strategy`, joinable by region for actual-vs-target comparisons.
- **Document management** — List and delete knowledge-base documents from the UI.
- **Multi-turn chat** — The last 6 messages are passed along as context.

### Tech stack

| Layer | Choice |
| --- | --- |
| Web framework | Flask |
| LLM | Zhipu GLM-4-Flash (`zhipuai` SDK) |
| Vector store | ChromaDB |
| Embedding / re-ranking | `BAAI/bge-small-zh-v1.5`, `BAAI/bge-reranker-base` |
| Keyword search | `rank_bm25` + `jieba` |
| Orchestration | LangGraph |
| Math / database | sympy, SQLite |

### Getting started

```bash
# 1. Install dependencies
pip install flask python-dotenv zhipuai PyPDF2 python-docx chromadb \
  langchain-text-splitters sentence-transformers jieba rank-bm25 langgraph sympy

# 2. Add your Zhipu API key
echo "ZHIPU_API_KEY=your-key" > .env

# 3. Build the sample database (required by Text2SQL)
python data/init_db.py

# 4. Run
python app.py          # then open http://127.0.0.1:5000
```

The first launch downloads the two BGE models (a few hundred MB). If the download is slow, set a mirror first:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### API

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/upload_knowledge` | Upload a document into the knowledge base (form-data, field `file`) |
| `POST` | `/ask` | Standard RAG question answering |
| `POST` | `/ask_agent` | Agent answering; also returns `sql_result`, `calc_result`, `retry_count` |
| `GET` | `/documents` | List documents in the knowledge base |
| `POST` | `/delete_document` | Delete a document by filename (JSON, field `filename`) |

### Configuration

Model and retrieval settings live in `config.py`: `CHAT_MODEL`, `CHUNK_SIZE` / `CHUNK_OVERLAP`, `TOP_K_VECTOR` / `TOP_K_BM25` / `TOP_K_RERANK`, `MAX_RETRY`, and `CHROMA_PATH`.

### Project layout

```
my-summarizer/
├── app.py                 # Flask entry point and routes
├── config.py              # Model and retrieval settings
├── rag_engine.py          # Chunking, vector search, BM25, RRF, re-ranking
├── agent/                 # LangGraph agent
│   ├── graph.py           #   State graph assembly
│   ├── nodes.py           #   plan / tool_plan / retrieve / evaluate / generate
│   ├── tools.py           #   Vector search / keyword search / calculator / Text2SQL
│   ├── state.py           #   AgentState definition
│   └── runner.py          #   run_agent entry point
├── sql_engine/            # Text2SQL
│   ├── schema.py          #   Table descriptions fed to the LLM
│   ├── text_to_sql.py     #   Natural language → SQL
│   ├── executor.py        #   Execution and result formatting
│   └── db.py              #   SQLite connection
├── data/init_db.py        # Generates the sample sales.db
├── templates/index.html   # Single-page frontend
└── static/                # main.js / style.css
```

---

## License / 许可

MIT License © 2026 188Y — see [LICENSE](LICENSE).
