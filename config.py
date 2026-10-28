import os

from dotenv import load_dotenv
from zhipuai import ZhipuAI

# 项目根目录：.env / chroma_db / data 等资源一律基于它定位，
# 保证从任意工作目录启动都能找到（原来相对 CWD 的写法换目录运行会静默丢库）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(BASE_DIR, ".env"))

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
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "documents"

# ============ 启动校验 ============
# key 缺失时在这里给出明确的修复指引，而不是等到各模块构造 ZhipuAI() 时
# 抛出令人摸不着头脑的裸 traceback（甚至整个 Flask 起不来）
if not ZHIPU_API_KEY:
    raise RuntimeError(
        "缺少 ZHIPU_API_KEY：请在项目根目录创建 .env 文件并写入一行 "
        "'ZHIPU_API_KEY=你的密钥'（智谱开放平台 https://open.bigmodel.cn 可获取）"
    )

# 全局共享的智谱客户端：app / rag_engine / agent / text_to_sql 统一复用，
# 避免各模块在 import 阶段各自构造（原来 key 缺失时会在 import 时直接崩）
zhipu_client = ZhipuAI(api_key=ZHIPU_API_KEY)
