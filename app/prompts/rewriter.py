REWRITE_PROMPT = """
You are an expert query rewriter.

The user's question failed to retrieve relevant documents from a vector
database. A grader inspected the retrieved chunks and gave these reasons
why they were irrelevant:

{grade_reasons}

Original question:
{question}

Rewrite the question to fix the problem identified above. Make it more
specific, add likely keywords, or shift the phrasing toward the actual
topic the user needs.

Only return the rewritten question, nothing else.
"""
