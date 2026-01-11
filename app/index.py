import os
import uuid
from dotenv import load_dotenv
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain.storage import LocalFileStore, create_kv_docstore

from app.pre_process import process_document

# ------------------------------------------------
# Load environment variables
# ------------------------------------------------
load_dotenv()


# ------------------------------------------------
# Assign stable IDs to parent documents
# ------------------------------------------------
def assign_ids(docs: List[Document]) -> List[Document]:
    for doc in docs:
        doc.metadata["doc_id"] = str(uuid.uuid4())
    return docs


# ------------------------------------------------
# Indexing pipeline (callable from API or CLI)
# ------------------------------------------------
def create_indexer_from_env() -> ParentDocumentRetriever:
    """
    Index document specified by DOCUMENT_PATH and return retriever.
    Safe to call from FastAPI.
    """

    file_path = os.getenv("DOCUMENT_PATH")

    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError("❌ DOCUMENT_PATH invalid")

    print(f"📄 Loading document: {file_path}")

    documents = assign_ids(process_document(file_path))
    print(f"🧹 Pages loaded: {len(documents)}")

    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL_NAME"),
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

    # ✅ Persistent docstore (SERIALIZES Document correctly)
    file_store = LocalFileStore("./data/parent_docs")
    docstore = create_kv_docstore(file_store)

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        parent_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=100
        ),
        child_splitter=RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=50
        ),
    )

    print("🔄 Indexing documents...")
    retriever.add_documents(documents)

    print(
        f"✅ Indexing complete. "
        f"Child chunks stored: {vectorstore._collection.count()}"
    )

    return retriever


# ------------------------------------------------
# Local test run ONLY
# ------------------------------------------------
if __name__ == "__main__":
    print("🔥 index module running in standalone mode")
    create_indexer_from_env()
