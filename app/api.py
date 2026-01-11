from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import shutil
import os
from typing import Dict

from app.index import create_indexer_from_env
from app.retriver import load_retriever, build_rag_chain
from app.evaluate import evaluate_with_gemini

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(
    title="Live RAG API",
    description="RAG with live indexing, retrieval, and Gemini evaluation",
    version="2.0.0",
)

# ---------------------------
# Static + Templates (UI)
# ---------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---------------------------
# Upload directory
# ---------------------------
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------------
# Models
# ---------------------------
class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    evaluation: Dict


class EvaluateRequest(BaseModel):
    question: str
    answer: str
    contexts: list[str]


# ---------------------------
# Health
# ---------------------------
@app.get("/")
def health():
    return {"status": "ready"}


# ---------------------------
# UI Route
# ---------------------------
@app.get("/ui", response_class=HTMLResponse)
def ui(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


# ---------------------------
# Index document
# ---------------------------
@app.post("/index")
def index_document(file: UploadFile = File(...)):
    """
    Upload + preprocess + index document
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Set env dynamically
    os.environ["DOCUMENT_PATH"] = file_path

    # Persistent indexing
    create_indexer_from_env()

    return {
        "message": "Document indexed successfully",
        "filename": file.filename,
    }


# ---------------------------
# Ask question (RAG + Evaluation)
# ---------------------------
@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    Stateless RAG inference + Gemini evaluation
    """

    # 1. Load retriever from persistent storage
    retriever = load_retriever()
    rag_chain = build_rag_chain(retriever)

    # 2. Retrieve context
    retrieved_docs = retriever.invoke(request.question)
    contexts = [doc.page_content for doc in retrieved_docs]

    # 3. Generate answer
    response = rag_chain.invoke(request.question)
    answer = response.content

    # 4. Evaluate (SAFE)
    try:
        evaluation = evaluate_with_gemini(
            question=request.question,
            answer=answer,
            contexts=contexts,
        )
    except Exception as e:
        evaluation = {
            "faithfulness_score": 0.0,
            "hallucination": "Unknown",
            "explanation": f"Evaluation failed: {str(e)}",
        }

    return {
        "answer": answer,
        "evaluation": evaluation,
    }


# ---------------------------
# Standalone evaluation endpoint
# ---------------------------
@app.post("/evaluate")
def evaluate_answer(request: EvaluateRequest):
    result = evaluate_with_gemini(
        question=request.question,
        answer=request.answer,
        contexts=request.contexts,
    )
    return {"evaluation": result}
