CONDENSE_PROMPT = """
You are managing a conversation between a user and a document Q&A assistant.

Given the chat history and the user's latest message, rewrite the latest
message as a STANDALONE question that can be understood without the chat
history. Resolve pronouns and references ("it", "that answer", "page 12")
using the history.

If the latest message is already self-contained, return it unchanged.

Recent chat history:
{history}

Latest user message:
{question}

Return ONLY the standalone question, nothing else.
"""


def format_history(history: list[dict], max_messages: int = 8) -> str:
    """Render recent chat history as 'Role: content' lines for prompts."""

    recent = [h for h in history if h.get("role") in ("user", "assistant")]
    recent = recent[-max_messages:]

    if not recent:
        return "(no previous conversation)"

    return "\n".join(
        f"{'User' if h['role'] == 'user' else 'Assistant'}: {h.get('content', '')}"
        for h in recent
    )
