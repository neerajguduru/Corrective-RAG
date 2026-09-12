from pathlib import Path
import shutil

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.graph.workflow import graph
from app.ingestion.ingest import IngestionPipeline

router = APIRouter()

pipeline = IngestionPipeline()

UPLOAD_DIR = Path("data/documents")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/")
def home():
    return {"message": "Corrective RAG API"}


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF supported")

    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    pipeline.ingest(str(file_path))

    return {
        "message": f"{file.filename} uploaded successfully."
    }


@router.post("/chat")
def chat(question: str):

    result = graph.invoke(
        {
            "question": question,
        }
    )

    return {
        "question": question,
        "grade": result.get("grade"),
        "rewritten_question": result.get("rewritten_question"),
        "rewrite_happened": result.get("rewritten_question") is not None,
        "answer": result["generation"]
    }


@router.get("/health")
def health():

    return {
        "status": "healthy"
    }