from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader


class DocumentLoader:
    """Loads supported documents."""

    @staticmethod
    def load(file_path: str) -> list[Document]:

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(file_path)

        suffix = path.suffix.lower()

        if suffix != ".pdf":
            raise ValueError("Currently only PDF files are supported.")

        loader = PyMuPDFLoader(file_path)

        return loader.load()