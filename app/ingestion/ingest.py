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

        logger.info(f"Loading {file_path}")

        documents = self.loader.load(file_path)

        chunks = self.splitter.split(documents)

        logger.info(f"Created {len(chunks)} chunks")

        vector_db.add_documents(chunks)

        logger.success(
            f"{Path(file_path).name} indexed successfully."
        )