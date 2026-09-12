from pathlib import Path

from app.llm.model import llm

from app.prompts.grader import GRADE_PROMPT
from app.prompts.generator import GENERATOR_PROMPT
from app.prompts.rewriter import REWRITE_PROMPT

from app.retrieval.retriever import Retriever


retriever = Retriever()


def retrieve(state):

    question = state["question"]
    selected_docs = state.get("selected_docs") or []

    print(f"\n[NODE] RETRIEVE — searching for: '{question}'")
    if selected_docs:
        print(f"[NODE] RETRIEVE — filtering to docs: {selected_docs}")

    docs = retriever.retrieve(
        question,
        doc_filter=selected_docs or None,
    )
    print(f"[NODE] RETRIEVE — found {len(docs)} chunks")

    return {
        "documents": docs
    }


def grade_documents(state):

    question = state["question"]
    docs = state["documents"]
    print(f"\n[NODE] GRADE — evaluating {len(docs)} chunks...")

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    prompt = GRADE_PROMPT.format(
        question=question,
        context=context,
    )

    response = llm.invoke(prompt)

    grade = response.content.strip().upper()
    print(f"[NODE] GRADE — verdict: {grade}")

    return {
        "grade": grade
    }


def rewrite_query(state):

    question = state["question"]
    selected_docs = state.get("selected_docs") or []
    print(f"\n[NODE] REWRITE — original: '{question}'")

    prompt = REWRITE_PROMPT.format(
        question=question
    )

    response = llm.invoke(prompt)

    rewritten = response.content.strip()
    print(f"[NODE] REWRITE — rewritten to: '{rewritten}'")

    docs = retriever.retrieve(
        rewritten,
        doc_filter=selected_docs or None,
    )
    print(f"[NODE] REWRITE — re-retrieved {len(docs)} chunks")

    return {
        "rewritten_question": rewritten,
        "documents": docs,
    }


def generate(state):

    question = state.get(
        "rewritten_question"
    ) or state["question"]

    docs = state["documents"]
    print(f"\n[NODE] GENERATE — answering with {len(docs)} chunks")

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    prompt = GENERATOR_PROMPT.format(
        question=question,
        context=context,
    )

    response = llm.invoke(prompt)
    print(f"[NODE] GENERATE — done\n")

    # Build source citations from retrieved chunks
    sources = []
    for doc in docs:
        meta = doc.metadata
        doc_name = meta.get("doc_name") or Path(
            meta.get("source", "unknown")
        ).name
        page = meta.get("page", 0)
        sources.append({
            "doc": doc_name,
            "page": int(page) + 1,          # 0-indexed → 1-indexed
            "snippet": doc.page_content[:300].strip(),
        })

    return {
        "generation": response.content,
        "sources": sources,
    }