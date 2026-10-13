# RAG Knowledge Assistant

[简体中文](README.md) | **English**

A Q&A demo combining **RAG retrieval**, a **LangGraph agent**, and **Text2SQL**: upload your own documents as a knowledge base, and it answers strictly from retrieved material — falling back to "not mentioned in the documents" when the sources don't cover it. When numbers are involved, it queries a SQLite database and does the math itself.

## Features

- **Knowledge-base Q&A** — Upload `.txt` / `.pdf` / `.docx`; the text is chunked and stored in ChromaDB. Answers are grounded strictly in retrieved chunks, never in the model's prior knowledge, and the assistant explicitly replies "not mentioned in the documents" when the sources fall short.
- **Hybrid retrieval** — LLM query rewriting → vector search (bge-small-zh) plus BM25 keyword search (jieba tokenizer) → RRF fusion → bge-reranker re-ranking, keeping the top 3 chunks.
- **Agent mode** — A LangGraph state machine `plan → tool_plan → retrieve → evaluate → generate`, where a failed evaluation loops back to `plan` (2 retries by default). Four tools are available: vector search, keyword search, a sympy calculator, and Text2SQL.
- **Text2SQL** — Ask questions in plain language against SQLite. Two sample tables ship with the repo: `sales` and `strategy`, joinable by region for actual-vs-target comparisons.
- **Document management** — List and delete knowledge-base documents from the UI.
- **Multi-turn chat** — The last 6 messages are passed along as context.

## Tech stack

| Layer | Choice |
| --- | --- |
| Web framework | Flask |
| LLM | Zhipu GLM-4-Flash (`zhipuai` SDK) |
| Vector store | ChromaDB |
| Embedding / re-ranking | `BAAI/bge-small-zh-v1.5`, `BAAI/bge-reranker-base` |
| Keyword search | `rank_bm25` + `jieba` |
| Orchestration | LangGraph |
| Math / database | sympy, SQLite |

## Getting started

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

## API

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/upload_knowledge` | Upload a document into the knowledge base (form-data, field `file`) |
| `POST` | `/ask` | Standard RAG question answering |
| `POST` | `/ask_agent` | Agent answering; also returns `sql_result`, `calc_result`, `retry_count` |
| `GET` | `/documents` | List documents in the knowledge base |
| `POST` | `/delete_document` | Delete a document by filename (JSON, field `filename`) |

## Configuration

Model and retrieval settings live in `config.py`: `CHAT_MODEL`, `CHUNK_SIZE` / `CHUNK_OVERLAP`, `TOP_K_VECTOR` / `TOP_K_BM25` / `TOP_K_RERANK`, `MAX_RETRY`, and `CHROMA_PATH`.

## Project layout

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

## License

MIT License © 2026 188Y — see [LICENSE](LICENSE).
