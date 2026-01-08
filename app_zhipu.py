# app_zhipu.py - PDF智能问答系统(智谱AI版本)
# 使用国内的智谱AI大模型

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import tempfile
from typing import List
import time

# PDF处理
try:
    from pypdf import PdfReader
except ImportError:
    print("请安装: pip install pypdf")

# 智谱AI
try:
    from zhipuai import ZhipuAI
except ImportError:
    print("请安装: pip install zhipuai")

# ============================================================================
# FastAPI应用初始化
# ============================================================================

app = FastAPI(
    title="PDF智能问答系统(智谱AI版)",
    description="使用智谱AI GLM-4的PDF问答系统",
    version="2.0-zhipu"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# 全局变量
# ============================================================================

pdf_text = ""
pdf_chunks = []
current_filename = None
zhipu_client = None

# ============================================================================
# 数据模型
# ============================================================================

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    model: str

class UploadResponse(BaseModel):
    status: str
    message: str
    filename: str
    chunks: int
    preview: str

# ============================================================================
# 核心功能函数
# ============================================================================

def init_zhipu_client():
    """初始化智谱AI客户端"""
    global zhipu_client
    
    api_key = os.getenv("ZHIPU_API_KEY")
    
    if not api_key:
        print("⚠️  警告: 未设置 ZHIPU_API_KEY")
        return False
    
    try:
        zhipu_client = ZhipuAI(api_key=api_key)
        print("✅ 智谱AI客户端初始化成功")
        return True
    except Exception as e:
        print(f"❌ 智谱AI初始化失败: {e}")
        return False

def extract_text_from_pdf(file_path: str):
    """从PDF提取文本"""
    print("[PDF] 开始提取文本...")
    reader = PdfReader(file_path)
    text = ""
    page_texts = []
    
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        text += page_text + "\n"
        page_texts.append((i+1, page_text))
    
    print(f"[PDF] 提取完成: {len(reader.pages)} 页, {len(text)} 字符")
    return text, page_texts

def split_text_with_metadata(text: str, chunk_size: int = 800):
    """
    分割文本并保留元数据
    chunk_size=800 因为中文大模型对中文更友好
    """
    print(f"[分割] 开始分割,块大小: {chunk_size}")
    chunks = []
    
    # 按段落分割
    paragraphs = text.split('\n\n')
    
    current_chunk = ""
    chunk_id = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append({
                    'id': chunk_id,
                    'text': current_chunk.strip(),
                    'length': len(current_chunk)
                })
                chunk_id += 1
            current_chunk = para + "\n\n"
    
    # 添加最后一块
    if current_chunk:
        chunks.append({
            'id': chunk_id,
            'text': current_chunk.strip(),
            'length': len(current_chunk)
        })
    
    print(f"[分割] 完成: {len(chunks)} 个文本块")
    return chunks

def simple_retrieval(question: str, chunks: List[dict], top_k: int = 3):
    """
    简单的检索算法
    真实场景应该用向量数据库,但为了简化先用关键词匹配
    """
    print(f"[检索] 搜索相关文本块, top_k={top_k}")
    
    question_lower = question.lower()
    question_words = set(question_lower.split())
    
    # 计算每个chunk的相关性分数
    scored_chunks = []
    for chunk in chunks:
        chunk_lower = chunk['text'].lower()
        chunk_words = set(chunk_lower.split())
        
        # 计算关键词重叠度
        overlap = len(question_words & chunk_words)
        
        # 计算包含度(问题词在chunk中出现)
        contains_count = sum(1 for word in question_words if word in chunk_lower)
        
        score = overlap * 2 + contains_count
        
        if score > 0:
            scored_chunks.append((score, chunk))
    
    # 按分数排序
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    
    # 返回top_k
    top_chunks = [chunk for score, chunk in scored_chunks[:top_k]]
    
    print(f"[检索] 找到 {len(top_chunks)} 个相关文本块")
    return top_chunks

def ask_zhipu(question: str, context_chunks: List[dict]):
    """
    调用智谱AI生成答案
    """
    global zhipu_client
    
    if not zhipu_client:
        return "错误: 智谱AI客户端未初始化", []
    
    # 构建上下文
    context = "\n\n---\n\n".join([
        f"文档片段 {i+1}:\n{chunk['text']}" 
        for i, chunk in enumerate(context_chunks)
    ])
    
    # 构建提示词
    prompt = f"""你是一个专业的PDF文档问答助手。请基于以下文档内容回答用户的问题。

【重要规则】
1. 只根据提供的文档内容回答
2. 如果文档中没有相关信息,请明确说"文档中未找到相关信息"
3. 不要编造文档中不存在的内容
4. 回答要准确、简洁、有条理
5. 用中文回答

【文档内容】
{context}

【用户问题】
{question}

【你的回答】
"""

    print(f"[智谱AI] 发送请求...")
    print(f"[智谱AI] 上下文长度: {len(context)} 字符")
    
    try:
        # 调用智谱AI
        response = zhipu_client.chat.completions.create(
            model="glm-4",  # 使用GLM-4模型
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,  # 降低随机性,更准确
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content
        
        print(f"[智谱AI] 回答生成成功")
        print(f"[智谱AI] 答案长度: {len(answer)} 字符")
        
        # 提取来源
        sources = [f"文档片段 {i+1}" for i in range(len(context_chunks))]
        
        return answer, sources
    
    except Exception as e:
        print(f"[智谱AI] ❌ 调用失败: {e}")
        return f"调用智谱AI失败: {str(e)}", []

# ============================================================================
# API端点
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    init_zhipu_client()

@app.get("/")
def read_root():
    return {
        "message": "🚀 PDF智能问答系统(智谱AI版)",
        "status": "running",
        "model": "GLM-4",
        "provider": "智谱AI",
        "current_document": current_filename,
        "zhipu_ready": zhipu_client is not None,
        "features": [
            "✅ PDF文本提取",
            "✅ 智能文本分割",
            "✅ 关键词检索",
            "✅ GLM-4智能问答",
            "✅ 完全中文支持"
        ]
    }

@app.get("/health")
def health_check():
    api_key = os.getenv("ZHIPU_API_KEY")
    return {
        "status": "healthy",
        "zhipu_api_configured": bool(api_key),
        "zhipu_client_ready": zhipu_client is not None,
        "document_loaded": current_filename is not None,
        "current_document": current_filename,
        "chunks_count": len(pdf_chunks)
    }

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    global pdf_text, pdf_chunks, current_filename
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="只支持PDF文件")
    
    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        print(f"\n{'='*60}")
        print(f"📄 处理PDF: {file.filename}")
        print(f"{'='*60}")
        
        # 提取文本
        pdf_text, page_texts = extract_text_from_pdf(tmp_path)
        
        # 分割文本
        pdf_chunks = split_text_with_metadata(pdf_text, chunk_size=800)
        
        current_filename = file.filename
        
        # 预览
        preview = pdf_text[:300] + "..." if len(pdf_text) > 300 else pdf_text
        
        print(f"{'='*60}")
        print("✅ 处理完成!可以提问了")
        print(f"{'='*60}\n")
        
        return UploadResponse(
            status="success",
            message=f"文档 '{file.filename}' 处理完成!共 {len(pdf_chunks)} 个文本块",
            filename=file.filename,
            chunks=len(pdf_chunks),
            preview=preview
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    if not pdf_chunks:
        raise HTTPException(status_code=400, detail="请先上传PDF文件")
    
    if not zhipu_client:
        raise HTTPException(
            status_code=500, 
            detail="智谱AI未配置,请设置 ZHIPU_API_KEY 环境变量"
        )
    
    try:
        print(f"\n{'='*60}")
        print(f"❓ 问题: {request.question}")
        print(f"{'='*60}")
        
        # Step 1: 检索相关文档
        relevant_chunks = simple_retrieval(request.question, pdf_chunks, top_k=3)
        
        if not relevant_chunks:
            return AnswerResponse(
                question=request.question,
                answer="抱歉,在文档中没有找到与您问题相关的内容。",
                sources=["无"],
                model="GLM-4"
            )
        
        # Step 2: 调用智谱AI生成答案
        answer, sources = ask_zhipu(request.question, relevant_chunks)
        
        print(f"{'='*60}")
        print("✅ 问答完成")
        print(f"{'='*60}\n")
        
        return AnswerResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            model="GLM-4"
        )
    
    except Exception as e:
        print(f"❌ 错误: {e}")
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")

@app.delete("/reset")
def reset_system():
    global pdf_text, pdf_chunks, current_filename
    
    old_file = current_filename
    pdf_text = ""
    pdf_chunks = []
    current_filename = None
    
    return {
        "status": "success",
        "message": f"已清除: {old_file}"
    }

# ============================================================================
# 启动配置
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("🚀 PDF智能问答系统 - 智谱AI版本启动")
    print("="*70)
    print("🤖 使用模型: GLM-4")
    print("🏢 提供商: 智谱AI (ChatGLM)")
    print("="*70)
    print("✨ 优势:")
    print("   ✅ 国内服务,速度快")
    print("   ✅ 中文能力强")
    print("   ✅ 价格便宜")
    print("   ✅ 稳定可靠")
    print("="*70)
    print("📖 API文档: http://localhost:8000/docs")
    print("❤️  健康检查: http://localhost:8000/health")
    print("="*70)
    print("⚙️  配置:")
    
    api_key = os.getenv("ZHIPU_API_KEY")
    if api_key:
        print(f"   ✅ ZHIPU_API_KEY: {api_key[:20]}...")
    else:
        print("   ⚠️  ZHIPU_API_KEY: 未设置")
        print("   请运行: set ZHIPU_API_KEY=你的密钥")
    
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)