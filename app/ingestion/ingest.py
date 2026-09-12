from pathlib import Path

from app.ingestion.loader import DocumentLoader
from app.ingestion.splitter import DocumentSplitter
from app.logger import logger
from app.retrieval.vectordb import vector_db


class IngestionPipeline:

    def __init__(self):

        self.loader = DocumentLoader()
        self.splitter = DocumentSplitter()

    def ingest(self, file_path: str):

        doc_name = Path(file_path).name

        # Remove old chunks for this doc to avoid duplicates on re-upload
        existing = vector_db._collection.get(
            where={"doc_name": {"$eq": doc_name}},
            include=[],
        )
        if existing["ids"]:
            logger.info(
                f"{doc_name} already indexed ({len(existing['ids'])} chunks). Replacing..."
            )
            vector_db._collection.delete(
                where={"doc_name": {"$eq": doc_name}}
            )

        logger.info(f"Loading {file_path}")

        documents = self.loader.load(file_path)

        # Tag every page with the doc name for filtering later
        for doc in documents:
            doc.metadata["doc_name"] = doc_name

        chunks = self.splitter.split(documents)

        logger.info(f"Created {len(chunks)} chunks")

        vector_db.add_documents(chunks)

        logger.success(
            f"{doc_name} indexed successfully."
        )