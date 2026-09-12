from langchain_chroma import Chroma

from app.config import CHROMA_DB
from app.retrieval.embeddings import embeddings


vector_db = Chroma(
    persist_directory=CHROMA_DB,
    embedding_function=embeddings,
)