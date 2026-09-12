import json
import shutil
import time
import traceback
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import BASE_DIR
from app.graph.workflow import graph
from app.ingestion.jobs import create_job, get_job, run_job
from app.ingestion.loader import DocumentLoader
from app.retrieval.retriever import Retriever
from app.telemetry import get_stats, log_event

router = APIRouter()

retriever_instance = Retriever()

UPLOAD_DIR = (BASE_DIR / "data" / "documents").resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    selected_docs: list[str] = Field(default_factory=list)
    # Conversation memory: [{"role": "user"|"assistant", "content": str}, ...]
    history: list[dict] = Field(default_factory=list)


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
    # Prevent path traversal: only allow bare filenames inside UPLOAD_DIR
    safe_name = Path(doc_name).name
    if safe_name != doc_name or doc_name in ("", ".", ".."):
        raise HTTPException(400, "Invalid document name.")

    deleted_chunks = retriever_instance.delete_document(safe_name)

    file_path = (UPLOAD_DIR / safe_name).resolve()
    if file_path.exists() and file_path.parent == UPLOAD_DIR.resolve():
        try:
            file_path.unlink()
        except Exception:
            pass

    return {
        "message": f"'{doc_name}' deleted successfully ({deleted_chunks} chunks removed).",
        "deleted_chunks": deleted_chunks,
    }


@router.post("/upload")
def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Save the file and schedule ingestion in the background.

    Returns immediately with a job_id — poll GET /jobs/{job_id} for status.
    Supports PDF, DOCX, TXT, MD.
    """

    filename = file.filename or ""
    safe_name = Path(filename).name
    suffix = Path(safe_name).suffix.lower()

    if (
        safe_name != filename
        or safe_name in ("", ".", "..")
        or suffix not in DocumentLoader.SUPPORTED_SUFFIXES
    ):
        raise HTTPException(
            400,
            f"Unsupported file. Supported types: "
            f"{sorted(DocumentLoader.SUPPORTED_SUFFIXES)}",
        )

    file_path = UPLOAD_DIR / safe_name

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except HTTPException:
        raise
    except Exception as e:
        # Failed to even save the file — clean up any partial write
        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(500, f"Failed to save file: {e}") from e

    job_id = create_job(safe_name)
    background_tasks.add_task(run_job, job_id, file_path)

    return {
        "job_id": job_id,
        "message": f"{safe_name} saved. Ingestion started in background.",
    }


@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Unknown job id.")
    return job


@router.post("/chat")
def chat(request: ChatRequest):
    start = time.monotonic()
    result = _run_graph(request)
    latency_ms = int((time.monotonic() - start) * 1000)

    summary = _summarize(request, result)

    log_event(
        "query",
        question=summary["question"],
        attempts=summary["attempts"],
        relevant_chunks=summary["relevant_chunks"],
        total_chunks=summary["total_chunks"],
        web_used=summary["web_used"],
        latency_ms=latency_ms,
    )

    return summary


@router.get("/stats")
def stats():
    """Aggregate branch telemetry: which loop branch fires per query,
    latency, fallback rate, and the index-health signal."""
    return get_stats()


def _state_defaults() -> dict:
    return {
        "rewritten_question": None,
        "attempts": 0,
        "web_used": False,
        "augmented": False,
        "refined_context": None,
        "chunk_grades": [],
        "verification": {"supported": True, "unsupported_claims": []},
        "verify_attempts": 0,
        "history": [],
    }


def _summarize(request: ChatRequest, result: dict) -> dict:
    chunk_grades = result.get("chunk_grades", [])

    return {
        "question": request.question,
        "active_question": result.get("question"),
        "rewritten_question": result.get("rewritten_question"),
        "rewrite_happened": result.get("rewritten_question") is not None,
        "attempts": result.get("attempts", 1),
        "relevant_chunks": sum(
            1 for g in chunk_grades if g.get("label") == "correct"
        ),
        "ambiguous_chunks": sum(
            1 for g in chunk_grades if g.get("label") == "ambiguous"
        ),
        "total_chunks": len(chunk_grades),
        "chunk_reasons": [g.get("reason", "") for g in chunk_grades],
        "web_used": result.get("web_used", False),
        "augmented": result.get("augmented", False),
        "refined": result.get("refined_context") is not None,
        "verified": (result.get("verification") or {}).get("supported", True),
        "unsupported_claims": (result.get("verification") or {}).get(
            "unsupported_claims", []
        ),
        "regenerated": (result.get("verify_attempts", 0) or 0) > 0,
        "sources": result.get("sources", []),
        "answer": result["generation"],
    }


def _run_graph(request: ChatRequest) -> dict:
    defaults = _state_defaults()
    defaults.pop("history", None)  # keep caller-supplied history
    return graph.invoke(
        {
            "question": request.question,
            "selected_docs": request.selected_docs,
            "history": request.history,
            **defaults,
        }
    )


@router.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """SSE streaming variant of /chat.

    Event sequence:
      meta   — grading outcome (chunk counts, rewrite status) before tokens
      token  — one chunk of the answer text (fallback + generate nodes only)
      done   — final metadata (sources, web_used, attempts)
    """

    return StreamingResponse(
        _chat_stream_events(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _chat_stream_events(request: ChatRequest):
    defaults = _state_defaults()
    defaults.pop("history", None)
    inputs = {
        "question": request.question,
        "selected_docs": request.selected_docs,
        "history": request.history,
        **defaults,
    }

    # Nodes whose streamed tokens are part of the visible answer
    ANSWER_NODES = {"generate", "fallback", "augment"}

    meta_sent = False
    collected: dict = {}
    start = time.monotonic()

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    try:
        for mode, payload in graph.stream(inputs, stream_mode=["updates", "messages"]):
            if mode == "updates":
                # payload: {node_name: state_patch}
                for patch in payload.values():
                    if not patch:
                        continue  # node returned no state changes
                    collected.update(patch)

                    relevant = sum(
                        1 for g in collected.get("chunk_grades", [])
                        if g.get("label") == "correct"
                    )
                    ambiguous = sum(
                        1 for g in collected.get("chunk_grades", [])
                        if g.get("label") == "ambiguous"
                    )
                    total = len(collected.get("chunk_grades", []))

                    if not meta_sent and ("chunk_grades" in patch or "web_used" in patch):
                        # Grade finished (or fallback started) — emit meta
                        meta = {
                            "relevant_chunks": relevant,
                            "ambiguous_chunks": ambiguous,
                            "total_chunks": total,
                            "rewrite_happened": collected.get("rewritten_question") is not None,
                            "rewritten_question": collected.get("rewritten_question"),
                            "attempts": collected.get("attempts", 1),
                            "web_used": collected.get("web_used", False),
                            "augmented": collected.get("augmented", False),
                        }
                        yield sse("meta", meta)
                        meta_sent = True

            elif mode == "messages":
                # payload: (message_chunk, metadata)
                chunk, metadata = payload
                node = metadata.get("langgraph_node", "")
                if node in ANSWER_NODES and chunk.content:
                    yield sse("token", {"text": chunk.content})

        final_relevant = sum(
            1 for g in collected.get("chunk_grades", [])
            if g.get("label") == "correct"
        )
        yield sse("done", {
            "sources": collected.get("sources", []),
            "web_used": collected.get("web_used", False),
            "augmented": collected.get("augmented", False),
            "refined": collected.get("refined_context") is not None,
            "verified": (collected.get("verification") or {}).get("supported", True),
            "unsupported_claims": (collected.get("verification") or {}).get(
                "unsupported_claims", []
            ),
            "regenerated": (collected.get("verify_attempts", 0) or 0) > 0,
            "attempts": collected.get("attempts", 1),
            "relevant_chunks": final_relevant,
            "ambiguous_chunks": sum(
                1 for g in collected.get("chunk_grades", [])
                if g.get("label") == "ambiguous"
            ),
            "total_chunks": len(collected.get("chunk_grades", [])),
            "latency_ms": int((time.monotonic() - start) * 1000),
        })

        log_event(
            "query",
            question=request.question,
            attempts=collected.get("attempts", 1),
            relevant_chunks=final_relevant,
            total_chunks=len(collected.get("chunk_grades", [])),
            web_used=collected.get("web_used", False),
            latency_ms=int((time.monotonic() - start) * 1000),
        )
    except Exception as e:
        traceback.print_exc()
        yield sse("error", {"message": str(e)})


@router.get("/health")
def health():
    return {"status": "healthy"}