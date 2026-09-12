from pathlib import Path

from app.ingestion.ingest import IngestionPipeline

pipeline = IngestionPipeline()

directory = Path("data/documents")

pdfs = list(directory.glob("*.pdf"))

if not pdfs:
    print("No PDF files found.")

for pdf in pdfs:

    pipeline.ingest(str(pdf))

print("Finished indexing.")