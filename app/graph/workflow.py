from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    augment,
    condense_question,
    fallback,
    filter_documents,
    generate,
    grade_documents,
    refine,
    regenerate_answer,
    retrieve,
    rewrite_query,
    route_after_grade,
    route_after_verify,
    verify_answer,
)
from app.graph.state import GraphState

builder = StateGraph(GraphState)

builder.add_node("condense", condense_question)
builder.add_node("retrieve", retrieve)
builder.add_node("grade", grade_documents)
builder.add_node("filter", filter_documents)
builder.add_node("refine", refine)
builder.add_node("rewrite", rewrite_query)
builder.add_node("augment", augment)
builder.add_node("fallback", fallback)
builder.add_node("generate", generate)
builder.add_node("verify", verify_answer)
builder.add_node("regenerate", regenerate_answer)

builder.add_edge(START, "condense")

builder.add_edge("condense", "retrieve")

builder.add_edge("retrieve", "grade")

# grade → filter (correct found) | augment (ambiguous) | rewrite (retry) | fallback (exhausted)
builder.add_conditional_edges(
    "grade",
    route_after_grade,
    {
        "correct": "filter",
        "ambiguous": "augment",
        "rewrite": "rewrite",
        "fallback": "fallback",
    },
)

# filter → refine (knowledge strips) → generate → verify → accept/retry
builder.add_edge("filter", "refine")
builder.add_edge("refine", "generate")
builder.add_edge("generate", "verify")

builder.add_conditional_edges(
    "verify",
    route_after_verify,
    {
        "regenerate": "regenerate",
        "accept": END,
    },
)

# regenerate calls back into verify (the retry answer is already complete)
builder.add_edge("regenerate", "verify")

# rewrite loops back to retrieve (graded again on the next pass)
builder.add_edge("rewrite", "retrieve")

# terminate permutations
builder.add_edge("augment", END)
builder.add_edge("fallback", END)

graph = builder.compile()
