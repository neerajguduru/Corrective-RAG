"""Batch-index every supported document in data/documents/.

Usage:
    python scripts/ingest_documents.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.loader import DocumentLoader
from app.ingestion.ingest import IngestionPipeline
from app.config import BASE_DIR

pipeline = IngestionPipeline()

directory = BASE_DIR / "data" / "documents"

files = sorted(
    f for f in directory.iterdir()
    if f.suffix.lower() in DocumentLoader.SUPPORTED_SUFFIXES
) if directory.exists() else []

if not files:
    print(f"No supported documents found in {directory}")
    raise SystemExit(0)

for path in files:
    print(f"\n=== {path.name} ===")
    try:
        pipeline.ingest(str(path))
    except Exception as e:
        print(f"FAILED: {e}")

print("\nFinished indexing.")
