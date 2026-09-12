from app.llm.model import llm

from app.prompts.grader import GRADE_PROMPT
from app.prompts.generator import GENERATOR_PROMPT
from app.prompts.rewriter import REWRITE_PROMPT

from app.retrieval.retriever import Retriever


retriever = Retriever()


def retrieve(state):

    question = state["question"]
    print(f"\n[NODE] RETRIEVE — searching for: '{question}'")

    docs = retriever.retrieve(question)
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
    print(f"\n[NODE] REWRITE — original: '{question}'")

    prompt = REWRITE_PROMPT.format(
        question=question
    )

    response = llm.invoke(prompt)

    rewritten = response.content.strip()
    print(f"[NODE] REWRITE — rewritten to: '{rewritten}'")

    docs = retriever.retrieve(rewritten)
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

    return {
        "generation": response.content
    }