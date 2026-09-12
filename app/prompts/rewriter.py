REWRITE_PROMPT = """
You are an expert query rewriter.

Rewrite the user's question so that it becomes easier for a vector database to retrieve relevant documents.

Only return the rewritten question.

Question:

{question}
"""