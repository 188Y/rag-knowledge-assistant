import os
from dotenv import load_dotenv
load_dotenv()

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from zhipuai import ZhipuAI
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import CHAT_MODEL

# 初始化 Rerank 模型
rerank_model = CrossEncoder('BAAI/bge-reranker-base')

# 全局缓存 BM25 索引
_bm25_index = None
_bm25_docs = None
_bm25_metas = None


def _invalidate_bm25_index():
    """文档库发生变化后调用，让下一次检索重建 BM25 索引"""
    global _bm25_index, _bm25_docs, _bm25_metas
    _bm25_index = None
    _bm25_docs = None
    _bm25_metas = None

# ============ 初始化本地 Embedding 模型 ============
# 首次运行会自动下载模型（约 100MB），之后使用本地缓存
# 如果下载慢，可以先在终端执行：export HF_ENDPOINT=https://hf-mirror.com
embedding_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

# 初始化智谱客户端（用于查询改写和回答生成）
zhipu_client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))  

# 初始化向量库（持久化到本地）
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
)

def get_embedding(text: str) -> list:
    """
    使用本地模型生成向量，完全免费，无需网络。
    normalize_embeddings=True 让向量归一化，配合余弦相似度效果更好。
    """
    embedding = embedding_model.encode(text, normalize_embeddings=True)
    return embedding.tolist()

def split_text(text: str) -> list:
    """将长文本切分为多个 chunk"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # 每个 chunk 约 500 字
        chunk_overlap=50,      # 相邻 chunk 重叠 50 字，避免语义断裂
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )
    return splitter.split_text(text)

def delete_document(doc_name: str) -> int:
    """根据文档名称从向量库中删除其所有片段"""
    all_data = collection.get(where={"source": doc_name})
    ids = all_data.get("ids", [])
    if ids:
        collection.delete(ids=ids)
        _invalidate_bm25_index()
    return len(ids)

def add_document(text: str, doc_name: str):
    """将文档切分、向量化后存入向量库（若已存在同名文档则先删除旧切片）"""
    delete_document(doc_name)
    chunks = split_text(text)
    
    ids = []
    documents = []
    embeddings = []
    metadatas = []
    
    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_name}_chunk_{i}"
        ids.append(chunk_id)
        documents.append(chunk)
        embeddings.append(get_embedding(chunk))
        metadatas.append({"source": doc_name, "chunk_index": i})
    
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    # 新增文档后旧索引已不完整，下次检索时重建
    _invalidate_bm25_index()
    
    return len(chunks)

def search(query: str, top_k: int = 3) -> list:
    """检索与问题最相关的 chunk"""
    query_embedding = get_embedding(query)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    # 整理检索结果
    retrieved = []
    for i in range(len(results["documents"][0])):
        retrieved.append({
            "content": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "distance": results["distances"][0][i]
        })
    
    return retrieved

def list_documents() -> list:
    """列出向量库中所有文档来源"""
    all_data = collection.get(include=["metadatas"])
    sources = set()
    for meta in all_data["metadatas"]:
        sources.add(meta["source"])
    return list(sources)

def rewrite_query(question: str) -> list:
    """用大模型把问题改写成多个检索 query"""
    prompt = f"""请把下面的问题改写成3个不同角度的检索查询，每行一个，不要编号：
问题：{question}"""
    
    response = zhipu_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    text = response.choices[0].message.content
    queries = [q.strip() for q in text.strip().split('\n') if q.strip()]
    return queries[:3]  # 最多取3个

def search_multi_query(question: str, top_k: int = 3) -> list:
    """多 query 检索，合并去重"""
    queries = [question] + rewrite_query(question)
    
    all_results = {}
    for q in queries:
        results = search(q, top_k=top_k)
        for r in results:
            # 用内容作为 key 去重
            key = r["content"][:50]
            if key not in all_results or r["distance"] < all_results[key]["distance"]:
                all_results[key] = r
    
    # 按距离排序
    return sorted(all_results.values(), key=lambda x: x["distance"])[:top_k * 2]

def build_bm25_index():
    """从向量库读取所有片段，构建 BM25 索引"""
    global _bm25_index, _bm25_docs, _bm25_metas
    
    all_data = collection.get(include=["documents", "metadatas"])
    docs = all_data["documents"]
    metas = all_data["metadatas"] or []
    
    if not docs:
        _bm25_index = False
        _bm25_docs = []
        _bm25_metas = []
        return
    
    # 中文分词
    tokenized = [list(jieba.cut(doc)) for doc in docs]
    _bm25_index = BM25Okapi(tokenized)
    _bm25_docs = docs
    # 保存 metadatas，否则检索结果无法还原真实文档名
    _bm25_metas = metas

def search_bm25(query: str, top_k: int = 5) -> list:
    """BM25 关键词检索"""
    if _bm25_index is None:
        build_bm25_index()
    if not _bm25_index:
        return []
    
    tokenized_query = list(jieba.cut(query))
    scores = _bm25_index.get_scores(tokenized_query)
    
    # 取 top_k
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    results = []
    seen = set()
    for i in top_indices:
        if scores[i] <= 0:
            continue
        # 不同文档可能含重复段落，同一内容只保留一次，否则 RRF 会被重复计分
        content = _bm25_docs[i]
        if content in seen:
            continue
        seen.add(content)
        
        meta = _bm25_metas[i] if _bm25_metas and i < len(_bm25_metas) else {}
        results.append({
            "content": content,
            "score": scores[i],
            "source": (meta or {}).get("source", "unknown")
        })
    return results

def rrf_fusion(vector_results: list, bm25_results: list, k: int = 60) -> list:
    """RRF 融合两路检索结果"""
    scores = {}
    content_map = {}
    
    # 向量检索结果
    for rank, r in enumerate(vector_results):
        key = r["content"][:50]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        content_map[key] = r
    
    # BM25 结果：同一片段只补充分数和 BM25 得分，
    # 不覆盖向量路已带好的 source，否则真实文档名会被检索器名顶掉
    for rank, r in enumerate(bm25_results):
        key = r["content"][:50]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        if key in content_map:
            content_map[key].setdefault("bm25_score", r.get("score"))
        else:
            content_map[key] = r
    
    # 按 RRF 分排序
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [content_map[key] for key in sorted_keys]

def rerank(query: str, candidates: list, top_k: int = 3) -> list:
    """对候选片段重排序"""
    if not candidates:
        return []
    
    # 构造 (query, passage) 对
    pairs = [(query, c["content"]) for c in candidates]
    scores = rerank_model.predict(pairs)
    
    # 按分数排序
    for i, c in enumerate(candidates):
        c["rerank_score"] = float(scores[i])
    
    sorted_candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return sorted_candidates[:top_k]

def search_enhanced(question: str, top_k: int = 3, queries: list = None) -> list:
    """阶段2的完整检索流程。

    queries 传入时直接复用已改写好的查询（如 Agent 的 plan_node 产出），
    不再调用 LLM 重复改写；缺省时保持原行为：现场调用 LLM 改写。
    """
    if queries:
        extra = [q for q in queries if q and q != question][:3]
        used_queries = list(dict.fromkeys([question] + extra))
    else:
        used_queries = [question] + rewrite_query(question)

    # 2. 多路检索
    vector_results = []
    for q in used_queries:
        vector_results.extend(search(q, top_k=5))
    
    # 3. BM25 检索
    bm25_results = search_bm25(question, top_k=5)
    
    # 4. RRF 融合
    fused = rrf_fusion(vector_results, bm25_results)
    
    # 5. 重排序
    final = rerank(question, fused[:20], top_k=top_k)
    
    return final