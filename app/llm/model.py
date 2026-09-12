from langchain_ollama import ChatOllama

from app.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL
)

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
)