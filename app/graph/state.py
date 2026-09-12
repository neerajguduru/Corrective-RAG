from typing import TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):

    question: str

    rewritten_question: str

    documents: list[Document]

    # Per-chunk grading results, aligned with `documents` by index:
    # [{"index": 0, "relevant": True, "reason": "..."}]
    chunk_grades: list[dict]

    generation: str

    sources: list[dict]

    selected_docs: list[str]

    # Conversation memory: [{"role": "user"|"assistant", "content": str}, ...]
    history: list[dict]

    # Loop control
    attempts: int                  # number of retrieve+grade passes done
    grade_reasons: list[str]       # why chunks were judged irrelevant (feeds rewriter)

    # Web fallback
    web_used: bool                 # True if web results contributed to the answer
    web_context: str               # text fetched from the web, if any

    # Three-tier CRAG state
    augmented: bool                # Ambiguous action: local + web merged
    refined_context: str           # verbatim knowledge strips (None = use chunks)

    # Answer verification (groundedness check)
    verification: dict             # {"supported": bool, "unsupported_claims": [...]}
    verify_attempts: int           # regeneration retries used
