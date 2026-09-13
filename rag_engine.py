import os
from dotenv import load_dotenv
load_dotenv()

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ============ 初始化本地 Embedding 模型 ============
# 首次运行会自动下载模型（约 100MB），之后使用本地缓存
# 如果下载慢，可以先在终端执行：export HF_ENDPOINT=https://hf-mirror.com
embedding_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

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

def add_document(text: str, doc_name: str):
    """将文档切分、向量化后存入向量库"""
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