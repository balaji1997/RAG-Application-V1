from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import shutil
import os
from typing import Any, Dict, List

from app.index import create_indexer_from_env
from app.retriver import (
    load_retriever,
    build_rag_chain,
    extract_sources,
)
from app.evaluate import evaluate_with_gemini

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(
    title="Live RAG API",
    description="RAG with live indexing, retrieval, evaluation, and source attribution",
    version="2.2.0",
)

# ---------------------------
# Static + Templates
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
    evaluation: Dict[str, Any]
    sources: List[Dict[str, Any]]


# ---------------------------
# Health
# ---------------------------
@app.get("/")
def health():
    return {"status": "ready"}


# ---------------------------
# UI
# ---------------------------
@app.get("/ui", response_class=HTMLResponse)
def ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ---------------------------
# Index document
# ---------------------------
@app.post("/index")
def index_document(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    os.environ["DOCUMENT_PATH"] = file_path
    create_indexer_from_env()

    return {"message": "Document Uploaded successfully"}


# ---------------------------
# Ask question (RAG + Evaluation + Conditional Sources)
# ---------------------------
@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):

    retriever = load_retriever()
    rag_chain = build_rag_chain(retriever)

    # 1. Retrieve
    retrieved_docs = retriever.invoke(request.question)
    contexts = [doc.page_content for doc in retrieved_docs]

    # 2. Generate answer
    response = rag_chain.invoke(request.question)
    answer = response.content.strip()

    # 3. Evaluate answer
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

    # 4. SOURCE GATING
    sources: List[Dict[str, Any]] = []

    answer_lower = answer.lower()
    irrelevant_phrases = [
        "i don't know",
        "cannot find",
        "could not find",
        "no relevant",
        "not provided in the context",
        "could not find any relevant information",
    ]

    if not any(phrase in answer_lower for phrase in irrelevant_phrases):
        sources = extract_sources(retrieved_docs)

    return {
        "answer": answer,
        "evaluation": evaluation,
        "sources": sources,  
    }
