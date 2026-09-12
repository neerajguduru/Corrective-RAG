from langchain_huggingface import HuggingFaceEmbeddings

from app.config import EMBEDDING_MODEL

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    },
)