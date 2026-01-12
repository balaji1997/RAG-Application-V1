import os
from dotenv import load_dotenv
from typing import List

# -------------------------------
# Reduce memory pressure (Windows)
# -------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from langchain_chroma import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.storage import LocalFileStore, create_kv_docstore

from app.evaluate import evaluate_with_gemini

load_dotenv()

# ------------------------------------------------
# Load Retriever
# ------------------------------------------------
def load_retriever() -> ParentDocumentRetriever:
    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv(
            "EMBEDDING_MODEL_NAME",
            "sentence-transformers/paraphrase-MiniLM-L3-v2",
        ),
        model_kwargs={"device": os.getenv("DEVICE", "cpu")},
        encode_kwargs={"batch_size": 8},
    )

    vectorstore = Chroma(
        collection_name="local_hf_pdr",
        embedding_function=embeddings,
        persist_directory=os.getenv(
            "PERSIST_DIRECTORY", "./data/chroma_db"
        ),
    )

    file_store = LocalFileStore("./data/parent_docs")
    docstore = create_kv_docstore(file_store)

    return ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        parent_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=100
        ),
        child_splitter=RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=50
        ),
    )


# ------------------------------------------------
# Helper: format retrieved docs (USED BY RAG)
# ------------------------------------------------
def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# ------------------------------------------------
# NEW: Helper to extract retrieval sources
# (Safe addition – does not affect frontend unless used)
# ------------------------------------------------
def extract_sources(docs: List[Document]) -> List[dict]:
    """
    Extracts human-readable source information
    for frontend display or logging.
    """
    sources = []
    for doc in docs:
        meta = doc.metadata or {}
        sources.append(
            {
                "source": os.path.basename(meta.get("source", "unknown")),
                "page": meta.get("page"),
                "snippet": doc.page_content[:300] + "...",
            }
        )
    return sources


# ------------------------------------------------
# Build Gemini RAG chain
# ------------------------------------------------
def build_rag_chain(retriever: ParentDocumentRetriever):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_template(
        """
Answer the question using the provided document context.
If the context is unrelated, say "Sorry, I could not find any relevant information in the provided document to answer this question".

Context:
{context}

Question:
{question}

Answer:
"""
    )

    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )


# ------------------------------------------------
# Local test run ONLY (unchanged behavior)
# ------------------------------------------------
if __name__ == "__main__":
    print("retriver module running in standalone mode")

    retriever = load_retriever()
    rag_chain = build_rag_chain(retriever)

    query = "What are the causes of climate change?"

    # ---------------------------
    # Retrieval
    # ---------------------------
    retrieved_docs = retriever.invoke(query)
    print(f"\n Retrieved {len(retrieved_docs)} documents")

    retrieved_contexts = [doc.page_content for doc in retrieved_docs]
    sources = extract_sources(retrieved_docs)

    # ---------------------------
    # Generation
    # ---------------------------
    response = rag_chain.invoke(query)
    answer = response.content

    print("\n Answer:\n")
    print(answer)

    # ---------------------------
    # Evaluation
    # ---------------------------
    print("\n Running Gemini-based evaluation...")

    evaluation = evaluate_with_gemini(
        question=query,
        answer=answer,
        contexts=retrieved_contexts,
    )

    print("\n Gemini Evaluation Result:\n")
    print(evaluation)

    print("\n Sources:\n")
    for s in sources:
        print(s)
