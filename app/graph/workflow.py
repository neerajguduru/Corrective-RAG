from langgraph.graph import START, END, StateGraph

from app.graph.state import GraphState

from app.graph.nodes import (
    retrieve,
    grade_documents,
    rewrite_query,
    generate,
)


builder = StateGraph(GraphState)

builder.add_node("retrieve", retrieve)
builder.add_node("grade", grade_documents)
builder.add_node("rewrite", rewrite_query)
builder.add_node("generate", generate)

builder.add_edge(START, "retrieve")

builder.add_edge("retrieve", "grade")


def route(state: GraphState):

    if state["grade"] == "YES":
        return "generate"

    return "rewrite"


builder.add_conditional_edges(
    "grade",
    route,
)

builder.add_edge("rewrite", "generate")

builder.add_edge("generate", END)

graph = builder.compile()