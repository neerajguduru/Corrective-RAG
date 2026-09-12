GRADE_PROMPT = """
You are a retrieval evaluator.

Question:
{question}

Retrieved chunks:
{chunks}

For EACH chunk, judge how well it helps answer the question using
exactly one of these labels:

- "correct"   — the chunk clearly contains the information needed to answer
- "ambiguous" — the chunk is related to the topic but does not sufficiently
                answer it (mentions the topic, buries the answer, or covers
                only part of the question)
- "incorrect" — the chunk is unrelated to the question

Respond with ONLY a JSON object in this exact shape:

{{"grades": [{{"chunk": 1, "label": "correct", "reason": "contains X"}}, {{"chunk": 2, "label": "incorrect", "reason": "discusses Y instead"}}]}}

Rules:
- One entry per chunk, numbered 1..{num_chunks}.
- "reason" must be at most 12 words.
- Only use "correct" when you are confident the answer is present.
"""
