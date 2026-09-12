VERIFY_PROMPT = """
You are an answer verifier.

Question:
{question}

Context the answer was generated from:
{context}

Answer to verify:
{answer}

Check whether every factual claim in the answer is supported by the
context above.

Respond with ONLY a JSON object in this exact shape:

{{"supported": true, "unsupported_claims": [...]}}

Rules:
- "supported" is true ONLY if every factual claim appears in (or is a
  faithful restatement of) the context.
- "unsupported_claims" lists the claims (as short phrases) that are NOT
  backed by the context. Empty list when supported.
- Be strict about numbers, names, and dates; lenient about phrasing.
"""


REGENERATE_PROMPT = """
You are a helpful AI research assistant.

Answer ONLY using the provided context. A previous answer was rejected
because it contained claims not supported by the context:

{flagged_claims}

Do NOT repeat those claims. If the answer is not in the context, say
"I don't know."

Context:

{context}

Question:

{question}
"""
