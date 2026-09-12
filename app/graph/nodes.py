import json
import re
from pathlib import Path

from app.config import (
    MAX_ATTEMPTS,
    MAX_VERIFY_ATTEMPTS,
    STRIPS_ENABLED,
    VERIFY_ENABLED,
    WEB_SEARCH,
)
from app.llm.model import json_llm, llm
from app.prompts.augment import AUGMENT_PROMPT
from app.prompts.condenser import CONDENSE_PROMPT, format_history
from app.prompts.generator import GENERATOR_PROMPT
from app.prompts.grader import GRADE_PROMPT
from app.prompts.refiner import REFINE_PROMPT
from app.prompts.rewriter import REWRITE_PROMPT
from app.prompts.verifier import REGENERATE_PROMPT, VERIFY_PROMPT
from app.prompts.web_generator import WEB_GENERATOR_PROMPT
from app.retrieval.retriever import Retriever
from app.retrieval.web_search import search_keywords, web_search
from app.telemetry import log_event

retriever = Retriever()


def condense_question(state):
    """Rewrite a follow-up question into a standalone question using chat
    history. No-op when there is no history."""

    question = state["question"]
    history = state.get("history") or []

    if not history:
        return {}

    print(f"\n[NODE] CONDENSE — follow-up: '{question}'")

    prompt = CONDENSE_PROMPT.format(
        history=format_history(history),
        question=question,
    )

    response = llm.invoke(prompt)
    standalone = response.content.strip()

    # Guards: empty or echoed rewrite → keep original
    if not standalone or standalone.lower() == question.lower():
        print("[NODE] CONDENSE — standalone unusable, keeping original")
        return {}

    # Safety: never let condensing balloon the question
    if len(standalone) > len(question) * 4 + 200:
        print("[NODE] CONDENSE — standalone too long, keeping original")
        return {}

    print(f"[NODE] CONDENSE — standalone: '{standalone}'")

    return {"question": standalone}


def _format_chunks(docs) -> str:
    return "\n\n".join(
        f"Chunk {i + 1}:\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )


def _build_sources(docs) -> list[dict]:
    sources = []
    for doc in docs:
        meta = doc.metadata
        doc_name = meta.get("doc_name") or Path(
            meta.get("source", "unknown")
        ).name
        page = meta.get("page", 0)
        sources.append({
            "doc": doc_name,
            "page": int(page) + 1,          # 0-indexed → 1-indexed
            "snippet": doc.page_content[:300].strip(),
        })
    return sources


def retrieve(state):

    attempts = state.get("attempts", 0)
    question = state["question"]
    selected_docs = state.get("selected_docs") or []

    print(f"\n[NODE] RETRIEVE (attempt {attempts + 1}) — searching for: '{question}'")
    if selected_docs:
        print(f"[NODE] RETRIEVE — filtering to docs: {selected_docs}")

    docs = retriever.retrieve(
        question,
        doc_filter=selected_docs or None,
    )
    print(f"[NODE] RETRIEVE — found {len(docs)} chunks")

    return {
        "documents": docs,
        "attempts": attempts + 1,
    }


def grade_documents(state):

    question = state["question"]
    docs = state["documents"]
    print(f"[NODE] GRADE — evaluating {len(docs)} chunks individually...")

    if not docs:
        print("[NODE] GRADE — no chunks retrieved")
        return {
            "chunk_grades": [],
            "grade_reasons": ["no chunks were retrieved for this query"],
        }

    prompt = GRADE_PROMPT.format(
        question=question,
        chunks=_format_chunks(docs),
        num_chunks=len(docs),
    )

    # json_llm uses Ollama's format="json" — output is guaranteed valid JSON
    response = json_llm.invoke(prompt)

    grades = _parse_chunk_grades(response.content, len(docs))

    correct = [g for g in grades if g["label"] == "correct"]
    ambiguous = [g for g in grades if g["label"] == "ambiguous"]
    reasons = [
        g["reason"]
        for g in grades
        if g["label"] in ("incorrect", "ambiguous") and g["reason"]
    ]

    print(
        f"[NODE] GRADE — {len(correct)} correct, {len(ambiguous)} ambiguous,"
        f" {len(grades) - len(correct) - len(ambiguous)} incorrect"
        f" (raw: {response.content.strip()[:80]})"
    )

    return {
        "chunk_grades": grades,
        "grade_reasons": reasons or ["retrieved chunks did not address the question"],
    }


def _parse_chunk_grades(raw: str, num_chunks: int) -> list[dict]:
    """Parse the grader's JSON into three labels; fall back to all-incorrect.

    Handles the clean shape as well as real-world LLM quirks observed with
    llama3.2: duplicated "grades" keys repeating one entry per object, entries
    with string chunk ids, or prose around the JSON.
    """

    text = raw.strip()

    # Strip markdown fences if the model wraps the JSON
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)

    # Real-world llama3.2 repeats `{"grades": [one entry]}` per chunk —
    # json.loads keeps only the LAST duplicate key, so scan entry-shaped
    # fragments directly first.
    parsed = _fragment_entries(text)

    if not parsed:
        parsed = _load_json_tolerant(text)
        # Accept both {"grades": [...]} and a bare [...] for robustness
        if isinstance(parsed, dict):
            parsed = parsed.get("grades", [])

    if not isinstance(parsed, list):
        parsed = []

    grades = []
    for i in range(num_chunks):
        entry = None
        for item in parsed:
            if isinstance(item, dict) and _chunk_id_matches(item.get("chunk"), i + 1):
                entry = item
                break

        if entry is None:
            # Model skipped a chunk — be safe and treat as incorrect
            grades.append(_fallback_grade(i, "not graded by evaluator"))
            continue

        grades.append(_normalize_grade(i, entry))

    return grades


def _chunk_id_matches(value, expected: int) -> bool:
    """Compare chunk ids tolerantly ("1"/1/'1 ' all match 1)."""

    if value is None:
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return int(value) == expected
    match = re.search(r"\d+", str(value))
    return bool(match) and int(match.group()) == expected


def _fragment_entries(text: str) -> list[dict]:
    """Collect per-chunk grade objects from fragmented LLM JSON output."""

    entries = []
    for m in re.finditer(r"\{[^{}]*\}", text):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if ("chunk" in obj or "relevant" in obj) and ("label" in obj or "reason" in obj):
            entries.append(obj)
    return entries


def _load_json_tolerant(text: str):
    """json.loads with fallbacks for fragmented LLM JSON output."""

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Salvage a bare array if present
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Fragmented fall-back: llama3.2 sometimes repeats {"grades": [one]}
    # objects. Extract every balanced non-nested {...} fragment individually.
    entries = []
    for obj_match in re.finditer(r"\{[^{}]*\}", text):
        frag = obj_match.group(0)
        try:
            obj = json.loads(frag)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("grades"), list):
            entries.extend(obj["grades"])

    if entries:
        return entries

    # Last resort: individual entry objects like {"chunk": 1, "label": ...}
    for obj_match in re.finditer(r"\{[^{}]*\}", text):
        try:
            obj = json.loads(obj_match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and ("chunk" in obj or "label" in obj or "relevant" in obj):
            entries.append(obj)

    return entries or None


def _normalize_grade(index: int, entry: dict) -> dict:
    """Normalize one grader entry into {"index", "label", "relevant", "reason"}.

    Supports the three-label schema ("label": correct/ambiguous/incorrect)
    and the legacy boolean schema ("relevant": bool) for robustness.
    """

    reason = str(entry.get("reason", "")).strip()[:200]
    raw_label = str(entry.get("label", "")).strip().lower()

    if raw_label in ("correct", "relevant"):
        label = "correct"
    elif raw_label == "ambiguous":
        label = "ambiguous"
    elif raw_label in ("incorrect", "irrelevant"):
        label = "incorrect"
    elif "relevant" in entry:
        # Legacy boolean shape — treat ambiguity as unknown → incorrect
        label = "correct" if entry.get("relevant") else "incorrect"
    else:
        label = "incorrect"

    return {
        "index": index,
        "label": label,
        "reason": reason,
    }


def _fallback_grade(index: int, reason: str) -> dict:
    return {"index": index, "label": "incorrect", "reason": reason, "relevant": False}


def filter_documents(state):
    """Keep only chunks the grader marked 'correct'."""

    docs = state["documents"]
    grades = state.get("chunk_grades", [])

    relevant_docs = [
        docs[g["index"]]
        for g in grades
        if g["label"] == "correct" and g["index"] < len(docs)
    ]

    print(f"[NODE] FILTER — kept {len(relevant_docs)}/{len(docs)} chunks")

    return {"documents": relevant_docs}


def route_after_grade(state) -> str:
    """CRAG three-action routing: Correct → refine, Ambiguous → augment,
    Incorrect → retry or fallback."""

    grades = state.get("chunk_grades", [])
    labels = {g["label"] for g in grades}

    if "correct" in labels:
        branch = "correct"
    elif "ambiguous" in labels:
        branch = "ambiguous"
    elif state.get("attempts", 0) >= MAX_ATTEMPTS:
        branch = "fallback"
    else:
        branch = "rewrite"

    log_event(
        "decision",
        branch=branch,
        question=state["question"],
        attempts=state.get("attempts", 0),
        relevant_chunks=sum(1 for g in grades if g["label"] == "correct"),
        total_chunks=len(grades),
    )

    return branch


def rewrite_query(state):

    question = state["question"]
    selected_docs = state.get("selected_docs") or []

    print(f"\n[NODE] REWRITE — original: '{question}'")

    reasons = state.get("grade_reasons") or []
    reasons_text = "\n".join(f"- {r}" for r in reasons) or "- no specific reason given"

    prompt = REWRITE_PROMPT.format(
        question=question,
        grade_reasons=reasons_text,
    )

    response = llm.invoke(prompt)

    rewritten = response.content.strip()

    # Guard against empty or echoed rewrites
    if not rewritten or rewritten.lower() == question.lower():
        rewritten = question
        print("[NODE] REWRITE — rewrite unusable, keeping original")

    print(f"[NODE] REWRITE — rewritten to: '{rewritten}'")

    docs = retriever.retrieve(
        rewritten,
        doc_filter=selected_docs or None,
    )
    print(f"[NODE] REWRITE — re-retrieved {len(docs)} chunks")

    return {
        "question": rewritten,          # becomes the active query for re-grading
        "rewritten_question": rewritten,
        "documents": docs,
    }


def fallback(state):

    """All retrieval attempts failed. Try web search, else fail honestly."""

    question = state["question"]
    print(f"\n[NODE] FALLBACK — documents exhausted after {state.get('attempts', 0)} attempts")

    if not WEB_SEARCH:
        print("[NODE] FALLBACK — web search disabled")
        return {
            "generation": (
                "I couldn't find information relevant to your question in the "
                "indexed documents. Try rephrasing, uploading a document that "
                "covers this topic, or enabling web fallback."
            ),
            "web_used": False,
            "sources": [],
        }

    print("[NODE] FALLBACK — searching the web...")
    keywords = search_keywords(question)
    web_context = web_search(keywords)

    if not web_context:
        print("[NODE] FALLBACK — web search returned nothing")
        return {
            "generation": (
                "I couldn't find information relevant to your question in the "
                "indexed documents, and a web search didn't return useful results."
            ),
            "web_used": False,
            "sources": [],
        }

    prompt = WEB_GENERATOR_PROMPT.format(
        question=question,
        context=web_context,
    )
    response = llm.invoke(prompt)
    print("[NODE] FALLBACK — web answer generated\n")

    return {
        "generation": response.content,
        "web_used": True,
        "augmented": False,
        "sources": [],
    }


def augment(state):
    """CRAG 'Ambiguous' action: chunks are related but not sufficient.
    Keep the ambiguous chunks AND fetch web results, merge both into
    one augmented context."""

    question = state["question"]
    docs = state["documents"]
    print(f"\n[NODE] AUGMENT — {len(docs)} ambiguous chunks + web search")

    web_context = ""
    if WEB_SEARCH:
        keywords = search_keywords(question)
        web_context = web_search(keywords)
        if web_context:
            print(f"[NODE] AUGMENT — fetched web context ({len(web_context)} chars)")

    local_context = "\n\n".join(doc.page_content for doc in docs)

    if not web_context:
        # No web results: ambiguous chunks are all we have
        context = local_context if local_context.strip() else ""
        if not context:
            return {
                "generation": (
                    "I couldn't find enough information to answer this "
                    "reliably, either in your documents or on the web."
                ),
                "web_used": False,
                "augmented": False,
                "sources": [],
            }
        prompt = GENERATOR_PROMPT.format(question=question, context=context)
        response = llm.invoke(prompt)
        print("[NODE] AUGMENT — WEB_SEARCH off/empty; answering from ambiguous chunks\n")
        return {
            "generation": response.content,
            "web_used": False,
            "augmented": True,
            "sources": _build_sources(docs),
        }

    prompt = AUGMENT_PROMPT.format(
        local_context=local_context or "(none)",
        web_context=web_context,
        question=question,
    )
    response = llm.invoke(prompt)
    print("[NODE] AUGMENT — combined local + web context\n")

    return {
        "generation": response.content,
        "web_used": True,
        "augmented": True,
        "sources": _build_sources(docs),
    }


def refine(state):
    """CRAG 'Correct' action refinement: decompose-then-recompose.
    Extract only the sentences that actually answer the question from the
    filtered (correct) chunks, verbatim. Falls back to full chunks when
    disabled or extraction yields nothing."""

    docs = state["documents"]
    question = state["question"]

    if not STRIPS_ENABLED or len(docs) > 6:
        # Gate: skip for very large contexts — cost/benefit flips
        if not STRIPS_ENABLED:
            print("[NODE] REFINE — strips disabled, using full chunks")
        else:
            print("[NODE] REFINE — too many chunks, using full chunks")
        return {"refined_context": None}

    print(f"\n[NODE] REFINE — extracting knowledge strips from {len(docs)} chunks")

    prompt = REFINE_PROMPT.format(
        question=question,
        chunks=_format_chunks(docs),
    )
    response = json_llm.invoke(prompt)

    sentences = _parse_refine_response(response.content)

    if not sentences:
        print("[NODE] REFINE — no usable strips, using full chunks")
        return {"refined_context": None}

    # Verify verbatim: strip must exist in the source chunks
    corpus = "\n\n".join(doc.page_content for doc in docs)
    valid = [s for s in sentences if s in corpus]
    dropped = len(sentences) - len(valid)

    if not valid:
        print("[NODE] REFINE — no verbatim matches, using full chunks")
        return {"refined_context": None}

    refined = "\n".join(valid)
    print(f"[NODE] REFINE — kept {len(valid)} sentences (dropped {dropped} non-verbatim)\n")

    log_event("refined", question=question)

    return {"refined_context": refined}


def _parse_refine_response(raw: str) -> list[str]:
    text = raw.strip()

    parsed = _load_json_tolerant(text)
    if isinstance(parsed, dict):
        parsed = parsed.get("sentences", [])

    if not isinstance(parsed, list):
        print(f"[NODE] REFINE — unparseable output: {raw[:120]}")
        return []

    return [s.strip() for s in parsed if isinstance(s, str) and s.strip()]


def generate(state):

    question = state["question"]
    docs = state["documents"]

    refined = state.get("refined_context")
    if refined:
        print(f"\n[NODE] GENERATE — answering with {len(refined)}-char refined strips")
        context = refined
    else:
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
    print("[NODE] GENERATE — done\n")

    return {
        "generation": response.content,
        "web_used": False,
        "augmented": False,
        "verification": {"supported": True, "unsupported_claims": []},
        "sources": _build_sources(docs),
    }


def verify_answer(state):
    """Check the generated answer against the context it came from.
    Unsupported claims trigger one regeneration retry, else surface as
    a low-confidence warning."""

    if not VERIFY_ENABLED:
        return {}

    question = state["question"]
    generation = state.get("generation", "")

    # Skip verifying non-answers ("I don't know") — nothing to check
    if "don't know" in generation.lower() or "couldn't answer" in generation.lower():
        return {}

    docs = state["documents"]
    refined = state.get("refined_context")
    context = refined if refined else "\n\n".join(d.page_content for d in docs)

    print(f"\n[NODE] VERIFY — checking answer against {len(context)}-char context")

    prompt = VERIFY_PROMPT.format(
        question=question,
        context=context,
        answer=generation,
    )
    response = json_llm.invoke(prompt)

    verdict = _parse_verdict(response.content)
    attempts = state.get("verify_attempts", 0)
    tries_used = attempts + 1

    print(f"[NODE] VERIFY — supported={verdict['supported']}"
          f" claims={verdict['unsupported_claims'][:3]}")

    log_event("verify", question=question, attempts=tries_used,
              branch="pass" if verdict["supported"] else "regenerate")

    out = {"verification": verdict}
    if not verdict["supported"]:
        # Increment only when a regeneration will actually follow
        out["verify_attempts"] = attempts + 1
    return out


def _parse_verdict(raw: str) -> dict:
    """Parse the verifier's JSON; unparseable output is treated as
    supported (do not punish the answer for our parser)."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)

    parsed = _load_json_tolerant(text)
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        parsed = parsed[0]  # fragmented single-object case
    if not isinstance(parsed, dict):
        print(f"[NODE] VERIFY — unparseable output: {text[:120]}")
        return {"supported": True, "unsupported_claims": []}

    claims = [
        str(c).strip()[:200]
        for c in parsed.get("unsupported_claims", [])
        if isinstance(c, str) and c.strip()
    ]
    supported = bool(parsed.get("supported", not claims))

    return {
        "supported": supported and not claims,
        "unsupported_claims": claims,
    }


def route_after_verify(state) -> str:
    """regenerate while retries remain and the answer is unsupported.

    verify_attempts counts regeneration-pending flags: first flag = 1,
    so `attempts <= MAX` allows exactly MAX_VERIFY_ATTEMPTS retries.
    """

    verdict = state.get("verification") or {}
    attempts = state.get("verify_attempts", 0)

    if (
        not verdict.get("supported", True)
        and attempts <= MAX_VERIFY_ATTEMPTS
    ):
        return "regenerate"

    return "accept"


def regenerate_answer(state):
    """Re-generate with a stricter prompt that names the flagged claims."""

    question = state["question"]
    docs = state["documents"]
    verdict = state.get("verification") or {}
    claims = "\n".join(f"- {c}" for c in verdict.get("unsupported_claims", []))

    context = state.get("refined_context") or "\n\n".join(
        doc.page_content for doc in docs
    )

    print(f"\n[NODE] REGENERATE — retrying with {len(verdict.get('unsupported_claims', []))}"
          " flagged claims removed")

    prompt = REGENERATE_PROMPT.format(
        flagged_claims=claims,
        context=context,
        question=question,
    )
    response = llm.invoke(prompt)
    print("[NODE] REGENERATE — done\n")

    return {
        "generation": response.content,
    }
