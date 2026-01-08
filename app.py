# app.py - PDF智能问答系统(简化启动版)
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

app = FastAPI(title="PDF智能问答系统")

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {
        "message": "🚀 PDF智能问答系统运行中!",
        "status": "running",
        "version": "1.0.0",
        "docs": "访问 /docs 查看API文档"
    }

@app.get("/health")
def health_check():
    openai_key = os.getenv("OPENAI_API_KEY")
    return {
        "status": "healthy",
        "openai_key_configured": bool(openai_key),
        "packages": {
            "fastapi": "installed",
            "langchain": "installed",
            "openai": "installed"
        }
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    上传PDF文件(演示版本)
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="只支持PDF文件")
    
    return {
        "status": "success",
        "message": f"文件 {file.filename} 上传成功!",
        "filename": file.filename,
        "note": "这是演示版本,完整功能正在开发中..."
    }

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    提问接口(演示版本)
    """
    return {
        "question": request.question,
        "answer": "这是一个演示回答。完整的RAG功能正在开发中...",
        "sources": ["演示数据"]
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("🚀 PDF智能问答系统启动中...")
    print("=" * 60)
    print("📖 API交互文档: http://localhost:8000/docs")
    print("❤️  健康检查:     http://localhost:8000/health")
    print("🏠 主页:         http://localhost:8000")
    print("=" * 60)
    print("💡 提示: 按 Ctrl+C 停止服务")
    print("=" * 60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)