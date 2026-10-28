import os
import glob

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import CHAT_MODEL, CHROMA_PATH, COLLECTION_NAME, zhipu_client


# ============ 模型加载：优先使用本地缓存，避免启动时联网卡死 ============
def _local_model_path(repo_dir_name: str, fallback_id: str) -> str:
    """
    优先返回 HuggingFace 本地缓存中的模型目录路径。
    好处：加载本地目录不会联网（HF 在线检查在国内网络下经常卡死在 SSL 握手）。
    本地没有缓存时回退为 repo id，由 sentence-transformers 联网下载（首次运行用）。
    """
    pattern = os.path.expanduser(f"~/.cache/huggingface/hub/{repo_dir_name}/snapshots/*/")
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else fallback_id


# 初始化 Rerank 模型（从本地缓存目录加载，离线可用）
rerank_model = CrossEncoder(
    _local_model_path("models--BAAI--bge-reranker-base", "BAAI/bge-reranker-base")
)

# BM25 索引采用不可变快照模式：重建时在局部变量里构建好完整的新对象，
# 最后一次性替换整个元组引用（CPython 的赋值是原子的），读侧无锁也不会
# 读到"半建索引"——比给三个全局变量分别加锁更简单也更快。
# 快照取值：None = 尚未构建；False = 空库；(BM25Okapi, docs, metas) = 可用
_bm25_snapshot = None


def _invalidate_bm25_index():
    """文档库发生变化后调用，让下一次检索重建 BM25 索引"""
    global _bm25_snapshot
    _bm25_snapshot = None

# ============ 初始化本地 Embedding 模型 ============
# 首次运行会自动下载模型（约 100MB），之后直接使用本地缓存目录，无需联网
embedding_model = SentenceTransformer(
    _local_model_path("models--BAAI--bge-small-zh-v1.5", "BAAI/bge-small-zh-v1.5")
)

# 初始化向量库（持久化到本地，路径基于项目根目录，换目录启动不会静默新建空库）
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
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

def documents_count() -> int:
    """知识库中的片段总数（供 Agent 空库短路等判断使用）"""
    return collection.count()

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
    global _bm25_snapshot

    all_data = collection.get(include=["documents", "metadatas"])
    docs = all_data["documents"]
    metas = all_data["metadatas"] or []

    if not docs:
        _bm25_snapshot = False
        return

    # 中文分词
    tokenized = [list(jieba.cut(doc)) for doc in docs]
    # 局部构建完成后整体替换引用：读侧要么拿到旧快照，要么拿到新快照，
    # 永远不会拿到 "新 index + 旧 docs" 这种混搭
    _bm25_snapshot = (BM25Okapi(tokenized), docs, metas)

def search_bm25(query: str, top_k: int = 5) -> list:
    """BM25 关键词检索"""
    snapshot = _bm25_snapshot
    if snapshot is None:
        build_bm25_index()
        snapshot = _bm25_snapshot
    if not snapshot:
        return []

    index, docs, metas = snapshot
    tokenized_query = list(jieba.cut(query))
    scores = index.get_scores(tokenized_query)

    # 取 top_k
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    seen = set()
    for i in top_indices:
        if scores[i] <= 0:
            continue
        # 不同文档可能含重复段落，同一内容只保留一次，否则 RRF 会被重复计分
        content = docs[i]
        if content in seen:
            continue
        seen.add(content)

        meta = metas[i] if metas and i < len(metas) else {}
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