"""In-memory registry for background ingestion jobs.

Jobs survive until process restart (fine for a local app). State flows:
queued -> running -> completed | failed
"""

import threading
import uuid
from pathlib import Path

from app.ingestion.ingest import IngestionPipeline
from app.telemetry import log_event

_lock = threading.Lock()
_jobs: dict[str, dict] = {}
_pipeline = IngestionPipeline()


def create_job(filename: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "filename": filename,
            "status": "queued",
            "message": "",
            "chunks": 0,
        }
    return job_id


def get_job(job_id: str) -> dict | None:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def _set(job_id: str, **fields) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)


def run_job(job_id: str, file_path: Path) -> None:
    """BackgroundTask entrypoint: ingest the file, update status."""
    _set(job_id, status="running")

    filename = file_path.name
    try:
        _pipeline.ingest(str(file_path))
        chunks = _count_chunks(filename)
        _set(job_id, status="completed", chunks=chunks,
             message=f"Indexed {chunks} chunks")
        log_event("job", branch=f"in::{path_suffix(filename)}",
                  question=filename, attempts=chunks)
    except Exception as e:
        _set(job_id, status="failed", message=str(e))
        # Remove the orphaned file so it doesn't linger un-indexed
        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            pass


def _count_chunks(filename: str) -> int:
    from app.retrieval.vectordb import vector_db

    result = vector_db._collection.get(
        where={"doc_name": {"$eq": filename}},
        include=[],
    )
    return len(result.get("ids", []))


def path_suffix(filename: str) -> str:
    return Path(filename).suffix.lstrip(".") or "unknown"
