from pathlib import Path
import shutil

from fastapi import APIRouter, UploadFile, File, HTTPException, Query

from app.graph.workflow import graph
from app.ingestion.ingest import IngestionPipeline
from app.retrieval.retriever import Retriever

router = APIRouter()

pipeline = IngestionPipeline()
retriever_instance = Retriever()

UPLOAD_DIR = Path("data/documents")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/")
def home():
    return {"message": "Corrective RAG API"}


@router.get("/documents")
def list_documents():
    """List all PDFs currently indexed in ChromaDB."""
    docs = retriever_instance.get_available_documents()
    return {"documents": docs}


@router.delete("/documents/{doc_name}")
def delete_document(doc_name: str):
    """Delete a document and all its chunks from ChromaDB and disk."""
    deleted_chunks = retriever_instance.delete_document(doc_name)

    file_path = UPLOAD_DIR / doc_name
    if file_path.exists():
        try:
            file_path.unlink()
        except Exception:
            pass

    return {
        "message": f"'{doc_name}' deleted successfully ({deleted_chunks} chunks removed).",
        "deleted_chunks": deleted_chunks,
    }


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF supported")

    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    pipeline.ingest(str(file_path))

    return {
        "message": f"{file.filename} uploaded and indexed successfully."
    }


@router.post("/chat")
def chat(
    question: str,
    selected_docs: list[str] = Query(default=[]),
):
    result = graph.invoke(
        {
            "question": question,
            "selected_docs": selected_docs,
        }
    )

    return {
        "question": question,
        "grade": result.get("grade"),
        "rewritten_question": result.get("rewritten_question"),
        "rewrite_happened": result.get("rewritten_question") is not None,
        "sources": result.get("sources", []),
        "answer": result["generation"],
    }


@router.get("/health")
def health():
    return {"status": "healthy"}