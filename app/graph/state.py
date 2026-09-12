from typing import TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):

    question: str

    rewritten_question: str

    documents: list[Document]

    generation: str

    grade: str

    sources: list[dict]           # NEW: source chunks used for the answer

    selected_docs: list[str]      # NEW: doc names to filter retrieval (empty = all)