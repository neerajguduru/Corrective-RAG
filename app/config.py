import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5"
)

_chroma_db_raw = os.getenv(
    "CHROMA_DB",
    str(BASE_DIR / "data" / "chroma_db"),
)

_chroma_db_path = Path(_chroma_db_raw)
if not _chroma_db_path.is_absolute():
    _chroma_db_path = BASE_DIR / _chroma_db_path

CHROMA_DB = str(_chroma_db_path)

TOP_K = int(
    os.getenv(
        "TOP_K",
        4
    )
)

MAX_ATTEMPTS = int(
    os.getenv(
        "MAX_ATTEMPTS",
        2
    )
)

WEB_SEARCH = os.getenv(
    "WEB_SEARCH",
    "true"
).lower() in ("true", "1", "yes")

# Hybrid retrieval: fetch this many candidates per retriever before RRF fusion
HYBRID_CANDIDATES = int(
    os.getenv(
        "HYBRID_CANDIDATES",
        10
    )
)

# Cross-encoder reranker
RERANK_ENABLED = os.getenv(
    "RERANK_ENABLED",
    "true"
).lower() in ("true", "1", "yes")

RERANK_MODEL = os.getenv(
    "RERANK_MODEL",
    "BAAI/bge-reranker-base"
)

# Keep this many chunks after reranking (fed to the grader)
RERANK_TOP_N = int(
    os.getenv(
        "RERANK_TOP_N",
        4
    )
)

# Sentence-level knowledge strips (CRAG decompose-then-recompose refinement)
STRIPS_ENABLED = os.getenv(
    "STRIPS_ENABLED",
    "true"
).lower() in ("true", "1", "yes")

# SQLite branch telemetry (data/telemetry.db)
TELEMETRY_ENABLED = os.getenv(
    "TELEMETRY_ENABLED",
    "true"
).lower() in ("true", "1", "yes")

# OCR for scanned/image-only PDFs (requires tesseract binary)
OCR_ENABLED = os.getenv(
    "OCR_ENABLED",
    "true"
).lower() in ("true", "1", "yes")

# Answer-groundedness verification + one regeneration retry
VERIFY_ENABLED = os.getenv(
    "VERIFY_ENABLED",
    "true"
).lower() in ("true", "1", "yes")

MAX_VERIFY_ATTEMPTS = int(
    os.getenv(
        "MAX_VERIFY_ATTEMPTS",
        1
    )
)