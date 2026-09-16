import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader
from docx import Document
from rag_engine import add_document, search_enhanced, list_documents
from agent import run_agent

load_dotenv()

app = Flask(__name__)
client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4"
)

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
    """基于知识库回答问题"""
    data = request.get_json()
    question = data.get('question', '').strip()
    
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
        
        # 3. 调用大模型生成回答
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[
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
                },
                {"role": "user", "content": question}
            ],
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

# ============ 原有路由保持 ============

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """原有的摘要接口，保持不变"""
    if 'file' not in request.files:
        return jsonify({"error": "没有文件上传"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "文件名为空"}), 400
    
    try:
        content = extract_text_from_file(file)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"文件解析失败: {str(e)}"}), 400
    
    summary = generate_summary(content)
    return jsonify({"summary": summary})


# ============ 新增路由：Agent 问答 ============
@app.route('/ask_agent', methods=['POST'])
def ask_agent():
    """基于 Agent 的问答接口"""
    data = request.get_json()
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({"error": "问题不能为空"}), 400
    
    try:
        result = run_agent(question)
        
        # 提取来源
        sources = list(set([
            d["source"] for d in result.get("documents", [])
            if d.get("source")
        ]))
        
        return jsonify({
            "answer": result["answer"],
            "sources": sources,
            "retry_count": result.get("retry_count", 0),
        })
    except Exception as e:
        return jsonify({"error": f"Agent 运行失败: {str(e)}"}), 500
    

def generate_summary(text):
    """原有的摘要函数，保持不变"""
    try:
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": "你是一个文档总结助手。请用简洁、清晰的中文，将用户提供的文本内容总结为200字以内的核心摘要。"},
                {"role": "user", "content": f"请总结以下内容：\n{text[:3000]}"}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"调用API出错: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)