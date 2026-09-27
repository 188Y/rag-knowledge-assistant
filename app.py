import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from zhipuai import ZhipuAI
from PyPDF2 import PdfReader
from docx import Document
from config import ZHIPU_API_KEY, CHAT_MODEL
from rag_engine import add_document, delete_document, search_enhanced, list_documents
from agent import run_agent

load_dotenv()

app = Flask(__name__)
client = ZhipuAI(api_key=ZHIPU_API_KEY)

# ============ 原有函数保持不变 ============

def extract_text_from_file(file):
    """根据文件类型提取纯文本"""
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == '.txt':
        return file.read().decode('utf-8')
    elif ext == '.pdf':
        return extract_from_pdf(file)
    elif ext == '.docx':
        return extract_from_docx(file)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")

def extract_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_from_docx(file):
    doc = Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# ============ 新增：RAG 相关接口 ============

@app.route('/upload_knowledge', methods=['POST'])
def upload_knowledge():
    """上传文档并存入知识库"""
    if 'file' not in request.files:
        return jsonify({"error": "没有文件上传"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "文件名为空"}), 400
    
    try:
        content = extract_text_from_file(file)
        chunk_count = add_document(content, file.filename)
        return jsonify({
            "message": f"文档已入库，共切分为 {chunk_count} 个片段",
            "filename": file.filename
        })
    except Exception as e:
        return jsonify({"error": f"入库失败: {str(e)}"}), 500

@app.route('/ask', methods=['POST'])
def ask():
    """基于知识库回答问题（支持多轮对话上下文）"""
    data = request.get_json()
    question = data.get('question', '').strip()
    history = data.get('history', [])
    
    if not question:
        return jsonify({"error": "问题不能为空"}), 400
    
    try:
        # 1. 检索相关片段
        retrieved = search_enhanced(question, top_k=3)
        
        if not retrieved:
            return jsonify({"answer": "知识库中还没有相关文档，请先上传。"})
        
        # 2. 拼接上下文
        context = "\n\n---\n\n".join([
            f"【来源：{r['source']}】\n{r['content']}" 
            for r in retrieved
        ])
        
        # 3. 构造请求消息（包含历史记录）
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个严格基于资料回答问题的助手。"
                    "请**只使用**下面提供的参考资料来回答问题。"
                    "如果参考资料中没有相关信息，必须明确回答'资料中未提及'，"
                    "绝对不要使用你自己的先验知识来补充或推断。"
                    "回答时请尽量引用原文表述。\n\n"
                    f"参考资料：\n{context}"
                )
            }
        ]
        # 拼接最近对话历史（最多取最近6条）
        if history and isinstance(history, list):
            for msg in history[-6:]:
                role = msg.get("role")
                content = (msg.get("content") or "").strip()
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})
        
        messages.append({"role": "user", "content": question})
        
        # 4. 调用大模型生成回答
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.3
        )
        
        answer = response.choices[0].message.content
        
        return jsonify({
            "answer": answer,
            "sources": [r["source"] for r in retrieved]
        })
    
    except Exception as e:
        return jsonify({"error": f"回答失败: {str(e)}"}), 500

@app.route('/documents', methods=['GET'])
def get_documents():
    """列出知识库中的所有文档"""
    try:
        docs = list_documents()
        return jsonify({"documents": docs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/delete_document', methods=['POST'])
def delete_doc():
    """从知识库中删除指定文档"""
    data = request.get_json() or {}
    filename = data.get('filename', '').strip()
    if not filename:
        return jsonify({"error": "文件名不能为空"}), 400
    try:
        count = delete_document(filename)
        return jsonify({"message": f"已删除文档 {filename}（清理了 {count} 个片段）"})
    except Exception as e:
        return jsonify({"error": f"删除失败: {str(e)}"}), 500

# ============ 原有路由保持 ============

@app.route('/')
def index():
    return render_template('index.html')

# ============ 新增路由：Agent 问答 ============
@app.route('/ask_agent', methods=['POST'])
def ask_agent():
    """基于 Agent 的问答接口（返回包含 SQL 数据、计算结果与执行状态的丰富信息）"""
    data = request.get_json()
    question = data.get('question', '').strip()
    history = data.get('history', [])
    
    if not question:
        return jsonify({"error": "问题不能为空"}), 400
    
    try:
        result = run_agent(question, history=history)
        
        # 提取来源
        sources = list(set([
            d["source"] for d in result.get("documents", [])
            if d.get("source")
        ]))
        
        return jsonify({
            "answer": result["answer"],
            "sources": sources,
            "retry_count": result.get("retry_count", 0),
            "sql_result": result.get("sql_result", {}),
            "calc_result": result.get("calc_result", ""),
            "tool_errors": result.get("tool_errors", []),
        })
    except Exception as e:
        return jsonify({"error": f"Agent 运行失败: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)