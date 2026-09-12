GRADE_PROMPT = """
You are a retrieval evaluator.

Question:

{question}

Retrieved Context:

{context}

Determine whether the retrieved context is sufficient to answer the question.

Reply ONLY with

YES

or

NO
"""