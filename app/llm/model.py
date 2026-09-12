from langchain_ollama import ChatOllama

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
    # Emit token callbacks even on .invoke() — required for SSE streaming
    streaming=True,
)

# JSON-mode instance: Ollama guarantees valid JSON output via `format`.
# Used for structured grading where parse failures break the loop.
json_llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
    format="json",
)
