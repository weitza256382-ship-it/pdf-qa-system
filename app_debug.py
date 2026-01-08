# app_debug.py - 调试版,看看PDF里到底有什么
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import tempfile
from typing import List

try:
    from pypdf import PdfReader
except ImportError:
    print("请安装: pip install pypdf")

try:
    from zhipuai import ZhipuAI
except ImportError:
    print("请安装: pip install zhipuai")

app = FastAPI(title="PDF问答系统(调试版)", version="debug")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
pdf_text = ""
pdf_chunks = []
current_filename = None
zhipu_client = None


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    model: str
    debug_info: dict  # 新增调试信息


class UploadResponse(BaseModel):
    status: str
    message: str
    filename: str
    chunks: int
    text_length: int  # 新增:文本长度
    first_200_chars: str  # 新增:前200个字符
    all_chunks_preview: List[str]  # 新增:所有块的预览


def init_zhipu_client():
    global zhipu_client
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        print("⚠️ ZHIPU_API_KEY 未设置")
        return False
    try:
        zhipu_client = ZhipuAI(api_key=api_key)
        print("✅ 智谱AI初始化成功")
        return True
    except Exception as e:
        print(f"❌ 智谱AI初始化失败: {e}")
        return False


def extract_text_from_pdf(file_path: str):
    print("[PDF] 提取文本...")
    reader = PdfReader(file_path)
    text = ""

    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        print(f"[PDF] 第{i+1}页: {len(page_text)} 字符")
        if page_text:
            print(f"[PDF] 第{i+1}页前50字: {page_text[:50]}")
        text += page_text + "\n\n"

    print(f"[PDF] 总共提取: {len(text)} 字符")
    return text


def split_text_simple(text: str, chunk_size: int = 800):
    """更简单的分割,确保有内容"""
    chunks = []

    # 如果文本很短,直接作为一个块
    if len(text) <= chunk_size:
        if text.strip():
            chunks.append({
                'id': 0,
                'text': text.strip(),
                'length': len(text)
            })
        return chunks

    # 按chunk_size分割
    for i in range(0, len(text), chunk_size):
        chunk_text = text[i:i + chunk_size].strip()
        if chunk_text:
            chunks.append({
                'id': len(chunks),
                'text': chunk_text,
                'length': len(chunk_text)
            })

    return chunks


def smart_retrieval(question: str, chunks: List[dict]):
    """改进的检索,更宽松的匹配"""
    if not chunks:
        return []

    print(f"[检索] 问题: {question}")
    print(f"[检索] 可用文本块: {len(chunks)}")

    # 如果只有一个块,直接返回
    if len(chunks) == 1:
        print("[检索] 只有1个块,直接使用")
        return chunks

    # 提取关键词
    question_lower = question.lower()
    keywords = [w for w in question_lower.split() if len(w) > 1]

    print(f"[检索] 关键词: {keywords}")

    # 评分
    scored = []
    for chunk in chunks:
        chunk_lower = chunk['text'].lower()
        score = sum(1 for kw in keywords if kw in chunk_lower)
        scored.append((score, chunk))
        print(f"[检索] 块{chunk['id']}: 得分={score}, 长度={chunk['length']}")

    # 排序
    scored.sort(reverse=True, key=lambda x: x[0])

    # 如果最高分是0,返回前3个块
    if scored[0][0] == 0:
        print("[检索] 没有匹配,返回前3个块")
        return [chunk for _, chunk in scored[:3]]

    # 返回得分>0的前3个块
    result = [chunk for score, chunk in scored if score > 0][:3]
    print(f"[检索] 返回 {len(result)} 个相关块\n")
    return result


def ask_zhipu_simple(question: str, context_chunks: List[dict]):
    """简化的智谱AI调用"""
    if not zhipu_client:
        return "智谱AI未初始化", [], {}

    # 构建上下文
    if not context_chunks:
        context = "(文档内容为空)"
    else:
        context = "\n\n".join([chunk['text'] for chunk in context_chunks])

    print(f"[智谱AI] 上下文长度: {len(context)} 字符")
    print(f"[智谱AI] 上下文预览: {context[:200]}...")

    prompt = f"""请根据以下文档内容回答问题。如果文档中确实没有相关信息,就说没有。
文档内容:
{context}
问题: {question}
回答:"""

    try:
        response = zhipu_client.chat.completions.create(
            model="glm-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800
        )

        answer = response.choices[0].message.content
        sources = [f"块{c['id']}" for c in context_chunks]

        debug = {
            "context_length": len(context),
            "chunks_used": len(context_chunks),
            "prompt_preview": prompt[:200]
        }

        return answer, sources, debug

    except Exception as e:
        print(f"[智谱AI] 错误: {e}")
        return f"调用失败: {str(e)}", [], {"error": str(e)}


@app.on_event("startup")
async def startup_event():
    init_zhipu_client()


@app.get("/")
def read_root():
    return {
        "message": "PDF问答系统(调试版)",
        "version": "debug",
        "current_document": current_filename,
        "chunks_count": len(pdf_chunks),
        "text_length": len(pdf_text)
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "zhipu_ready": zhipu_client is not None,
        "document_loaded": current_filename is not None,
        "current_document": current_filename,
        "chunks_count": len(pdf_chunks),
        "total_text_length": len(pdf_text)
    }


@app.get("/debug/text")
def get_full_text():
    """查看提取的完整文本和所有分块"""
    return {
        "filename": current_filename,
        "length": len(pdf_text),
        "text": pdf_text,
        "chunks": pdf_chunks
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    global pdf_text, pdf_chunks, current_filename

    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="只支持PDF")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        print(f"\n{'='*60}")
        print(f"处理文件: {file.filename}")
        print(f"{'='*60}")

        # 提取文本
        pdf_text = extract_text_from_pdf(tmp_path)

        print(f"[分割] 文本总长度: {len(pdf_text)}")

        # 分块
        pdf_chunks = split_text_simple(pdf_text, chunk_size=800)

        print(f"[分割] 分割结果: {len(pdf_chunks)} 个块")
        for chunk in pdf_chunks:
            print(f"  块{chunk['id']}: {chunk['length']} 字符")

        current_filename = file.filename

        return UploadResponse(
            status="success",
            message=f"处理完成，共 {len(pdf_chunks)} 个块",
            filename=file.filename,
            chunks=len(pdf_chunks),
            text_length=len(pdf_text),
            first_200_chars=pdf_text[:200] if pdf_text else "(空)",
            all_chunks_preview=[
                f"块{c['id']}: {c['text'][:100]}..."
                for c in pdf_chunks
            ]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    if not pdf_chunks:
        raise HTTPException(status_code=400, detail="请先上传PDF")

    if not zhipu_client:
        raise HTTPException(status_code=500, detail="智谱AI未配置，请检查环境变量 ZHIPU_API_KEY")

    try:
        print(f"\n{'='*60}")
        print(f"问题: {request.question}")
        print(f"{'='*60}")

        # 检索相关块
        relevant = smart_retrieval(request.question, pdf_chunks)

        # 调用大模型生成答案
        answer, sources, debug = ask_zhipu_simple(request.question, relevant)

        return AnswerResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            model="GLM-4",
            debug_info=debug
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("🔧 PDF问答系统 - 调试版启动")
    print("="*70)
    print("新增功能:")
    print(" - GET /debug/text    : 查看提取的完整文本和所有分块")
    print(" - 详细控制台日志输出")
    print(" - 更宽松的关键词检索")
    print(" - 上传后返回分块预览信息")
    print("="*70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)