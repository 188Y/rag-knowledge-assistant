import os
from dotenv import load_dotenv

load_dotenv()

# ============ API 配置 ============
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

# ============ 模型配置 ============
CHAT_MODEL = "glm-4-flash"           # 对话模型
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"   # 本地 embedding
RERANK_MODEL = "BAAI/bge-reranker-base"      # 本地 rerank

# ============ 检索参数 ============
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_VECTOR = 5       # 向量检索取 Top-5
TOP_K_BM25 = 5         # BM25 检索取 Top-5
TOP_K_RERANK = 3       # 重排序后取 Top-3
MAX_RETRY = 2          # Agent 最多重试次数

# ============ 向量库配置 ============
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "documents"