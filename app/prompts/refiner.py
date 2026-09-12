REFINE_PROMPT = """
You are a context distiller.

The question being answered:
{question}

The candidate context chunks:
{chunks}

From the chunks above, extract ONLY the sentences that are directly
needed to answer the question. Copy sentences VERBATIM — do not rewrite,
summarize, merge, or invent anything.

Respond with ONLY a JSON object in this exact shape:

{{"sentences": ["verbatim sentence 1", "verbatim sentence 2"]}}

Rules:
- Only include sentences copied exactly from the chunks.
- If nothing helps, return an empty list.
"""
