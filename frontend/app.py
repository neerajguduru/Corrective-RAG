import json
import os
import time

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Number of past (user, assistant) turns sent for follow-up questions
HISTORY_TURNS = 3

st.set_page_config(
    page_title="Corrective RAG Assistant",
    page_icon="📚",
    layout="wide",
)

# ── Session State Initialization (Must be first) ──────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


def clear_chat():
    st.session_state.messages = []


def _recent_history() -> list[dict]:
    """Last HISTORY_TURNS user/assistant exchanges for follow-up condensing."""
    eligible = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m["role"] in ("user", "assistant") and m.get("content")
    ]
    return eligible[-(HISTORY_TURNS * 2):]


def _parse_sse(response):
    """Yield (event, data_dict) tuples from a requests SSE stream."""
    event = None
    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        line = raw_line.strip()
        if not line:
            event = None
            continue
        if line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_str = line[len("data:"):].strip()
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                data = {"raw": data_str}
            yield event or "message", data


def _build_chat_payload(question: str) -> dict:
    return {
        "question": question,
        "selected_docs": selected_docs,
        "history": _recent_history(),
    }


def _run_background_ingest(files):
    """Upload files, then poll their ingestion jobs until they settle."""
    jobs: dict[str, str] = {}  # filename -> job_id

    for f in files:
        try:
            suffix = f.type or "application/octet-stream"
            response = requests.post(
                f"{API_URL}/upload",
                files={"file": (f.name, f, suffix)},
                timeout=60,
            )
            if response.ok:
                jobs[f.name] = response.json().get("job_id", "")
            else:
                st.error(f"{f.name}: {response.json().get('detail', 'upload failed')}")
        except Exception as e:
            st.error(f"{f.name}: connection error ({e})")

    if not jobs:
        return

    statuses = {name: ("queued", "") for name in jobs}
    placeholder = st.empty()

    with placeholder.container():
        while True:
            lines = []
            all_done = True
            for name, job_id in jobs.items():
                try:
                    r = requests.get(f"{API_URL}/jobs/{job_id}", timeout=5)
                    status = r.json()
                    state = status.get("status", "unknown")
                    msg = status.get("message", "")
                except Exception:
                    state, msg = "unknown", "poll failed"

                statuses[name] = (state, msg)
                icon = {
                    "completed": "✅", "failed": "❌",
                    "running": "⚙️", "queued": "⏳",
                }.get(state, "…")
                lines.append(f"{icon} **{name}** — {state} {msg}")
                if state not in ("completed", "failed"):
                    all_done = False

            st.markdown("  \n".join(lines))
            if all_done:
                break
            time.sleep(1)

    placeholder.empty()

    for name, (state, _) in statuses.items():
        if state == "completed":
            st.toast(f"Indexed {name}!", icon="✅")
        elif state == "failed":
            st.toast(f"Failed to index {name}", icon="❌")

    if any(state == "completed" for state, _ in statuses.values()):
        time.sleep(0.5)
        st.rerun()


def _render_assistant_message(msg, show_content=True):
    """Render an assistant message: badges, answer, citations.

    show_content=False on the live path (answer was already streamed).
    """
    relevant = msg.get("relevant_chunks", 0)
    total = msg.get("total_chunks", 0)
    attempts = msg.get("attempts", 1)
    rewrite_happened = msg.get("rewrite_happened", False)
    rewritten_q = msg.get("rewritten_question")
    web_used = msg.get("web_used", False)
    augmented = msg.get("augmented", False)
    refined = msg.get("refined", False)
    verified = msg.get("verified", True)
    regenerated = msg.get("regenerated", False)
    sources = msg.get("sources", [])

    cols = st.columns([1, 1, 2])
    with cols[0]:
        if augmented and web_used:
            st.markdown("**:violet[● Ambiguous + 🌐 web merged]**")
        elif web_used:
            st.markdown("**:violet[🌐 Web Fallback]**")
        elif relevant > 0:
            st.markdown(f"**:green[● {relevant}/{total} chunks relevant]**")
        else:
            st.markdown(f"**:orange[● 0/{total} chunks relevant]**")
    with cols[1]:
        extra = []
        if rewrite_happened:
            extra.append(f"🔄 rewritten ({attempts} tries)")
        if refined:
            extra.append("✂️ refined")
        if not verified:
            extra.append("⚠️ low confidence")
        elif regenerated:
            extra.append("✅ verified")
        if not extra:
            st.markdown("**:gray[➡️ Direct Retrieval]**")
        else:
            st.markdown("**:blue[" + " · ".join(extra) + "]**")

    if not verified:
        claims = msg.get("unsupported_claims") or []
        warn = "⚠️ The answer verifier flagged it as not fully grounded."
        if claims:
            warn += " Uncorroborated claims: " + "; ".join(claims[:3]) + "."
        st.warning(warn)

    if rewrite_happened and rewritten_q:
        st.caption(f"**Refined search query:** *{rewritten_q}*")

    if show_content:
        st.markdown(msg["content"])

    if sources:
        with st.expander(f"📑 View {len(sources)} Source Citations", expanded=False):
            for idx, src in enumerate(sources, 1):
                with st.container(border=True):
                    st.markdown(f"**{idx}. {src['doc']}** · *Page {src['page']}*")
                    st.caption(src["snippet"] + ("..." if len(src["snippet"]) >= 300 else ""))


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 Corrective RAG")
    st.caption("Self-evaluating document Q&A · LangGraph + Llama 3.2")
    st.divider()

    # 1. Document Management
    st.subheader("📂 Document Library")
    try:
        docs_response = requests.get(f"{API_URL}/documents", timeout=3)
        available_docs = docs_response.json().get("documents", [])
    except Exception:
        available_docs = []

    if available_docs:
        selected_docs = st.multiselect(
            "Filter Search Scope",
            options=available_docs,
            default=[],
            help="Select specific documents to query. Leave empty to search all documents.",
            placeholder="Searching all documents",
        )

        with st.expander(f"Manage Documents ({len(available_docs)})", expanded=False):
            for doc in available_docs:
                col_name, col_del = st.columns([3, 1])
                with col_name:
                    st.caption(f"📄 {doc}")
                with col_del:
                    if st.button("🗑️", key=f"del_{doc}", help=f"Delete {doc} from vector store"):
                        try:
                            del_res = requests.delete(f"{API_URL}/documents/{doc}", timeout=10)
                            if del_res.ok:
                                st.toast(f"Deleted {doc}", icon="🗑️")
                                st.rerun()
                            else:
                                st.error("Delete failed")
                        except Exception as e:
                            st.error(f"Error: {e}")
    else:
        st.info("No documents indexed yet. Upload a PDF below.")
        selected_docs = []

    st.divider()

    # 2. Upload Section (multi-file, background ingestion with status polling)
    st.subheader("⬆️ Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="Upload documents to chunk and embed into ChromaDB. "
             "Scanned PDFs are OCR'd automatically.",
        key="file_uploader",
    )

    if uploaded_files:
        if st.button(f"Index {len(uploaded_files)} document(s)", type="secondary", use_container_width=True):
            _run_background_ingest(uploaded_files)

    st.divider()

    # 3. Actions & Status
    st.button("🗑️ Clear Chat History", on_click=clear_chat, use_container_width=True)

    # 4. Index Health (branch telemetry)
    try:
        stats_response = requests.get(f"{API_URL}/stats", timeout=3)
        stats = stats_response.json()
    except Exception:
        stats = {}

    if stats and not stats.get("error"):
        with st.expander("📊 Index Health", expanded=False):
            st.caption(f"Queries answered: **{stats.get('total_queries', 0)}**")
            branch_counts = stats.get("branch_counts", {})
            if branch_counts:
                st.caption(
                    f"Relevant: **{branch_counts.get('correct', 0)}** · "
                    f"Ambiguous: **{branch_counts.get('ambiguous', 0)}** · "
                    f"Rewrite: **{branch_counts.get('rewrite', 0)}** · "
                    f"Fallback: **{branch_counts.get('fallback', 0)}**"
                )
            latency = stats.get("avg_latency_ms")
            if latency:
                st.caption(f"Avg latency: **{latency:.0f} ms**")
            refined = stats.get("refinement_count", 0)
            st.caption(f"Distilled answers (tiny details trimmed): **{refined}**")
            health = stats.get("index_health", "no_data")
            if health == "needs_work":
                st.warning(
                    "Over 25% of queries fail retrieval or trigger fallbacks. "
                    "Focus on chunking/embedding quality, not the correction loop."
                )
            elif health == "healthy":
                st.success("Retrieval is healthy.")
    st.caption("Backend: FastAPI on `localhost:8000`  \nModel: Llama 3.2 via Ollama")

# ── Main Content Area ─────────────────────────────────────────────────────────
st.header("Document Q&A Assistant")
st.caption("Corrective RAG validates retrieved chunks and automatically rewrites the query if context is insufficient.")

# ── Welcome Screen if no messages ─────────────────────────────────────────────
if not st.session_state.messages:
    with st.container(border=True):
        st.markdown("### 👋 Welcome!")
        st.markdown(
            "Ask questions against your uploaded documents. The system will:\n"
            "- **Retrieve** relevant chunks from ChromaDB\n"
            "- **Evaluate** context relevance using Llama 3.2\n"
            "- **Self-Correct** by rewriting queries if retrieval quality is low\n"
            "- **Generate** grounded answers with exact source citations"
        )
        if available_docs:
            st.caption(f"Currently indexed: {', '.join(available_docs)}")

# ── Render Message History ────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            _render_assistant_message(msg)

# ── Chat Input ────────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask a question about your documents...")

if prompt:
    # 1. Display and record user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 2. Assistant response — streamed token-by-token via SSE
    with st.chat_message("assistant"):
        with st.status("Evaluating retrieval & generating answer...", expanded=True) as status_box:
            try:
                response = requests.post(
                    f"{API_URL}/chat/stream",
                    json=_build_chat_payload(prompt),
                    stream=True,
                    timeout=180,
                )
                response.raise_for_status()
            except Exception as e:
                status_box.update(label="Error occurred", state="error", expanded=True)
                st.error(f"Failed to generate answer: {e}")
                st.stop()

        answer_parts: list[str] = []
        meta: dict = {}
        final: dict = {}

        # Stream answer tokens into the chat area as they arrive
        def _token_stream():
            global meta, final
            for event, data in _parse_sse(response):
                if event == "meta":
                    meta = data
                    relevant = data.get("relevant_chunks", 0)
                    ambiguous = data.get("ambiguous_chunks", 0)
                    total = data.get("total_chunks", 0)
                    if data.get("augmented"):
                        status_box.write(
                            f"🔀 {relevant}/{total} correct, {ambiguous} ambiguous — "
                            "merging ambiguous chunks with web results..."
                        )
                    elif data.get("web_used"):
                        status_box.write("🌐 Not in your documents — using web fallback...")
                    elif data.get("rewrite_happened"):
                        status_box.write(
                            f"🔄 0/{total} relevant → rewrote to "
                            f"*{data.get('rewritten_question')}* (attempt {data.get('attempts', 1)})"
                        )
                    else:
                        status_box.write(f"✅ {relevant}/{total} chunks relevant — generating...")
                    status_box.update(label="Generating answer...", state="running")
                elif event == "token":
                    answer_parts.append(data.get("text", ""))
                    yield data.get("text", "")
                elif event == "done":
                    final = data
                    status_box.update(label="Complete!", state="complete", expanded=False)
                    return
                elif event == "error":
                    status_box.update(label="Error occurred", state="error", expanded=True)
                    st.error(f"Backend error: {data.get('message', 'unknown')}")
                    return

        st.write_stream(_token_stream())

        msg = {
            "role": "assistant",
            "content": "".join(answer_parts),
            "relevant_chunks": meta.get("relevant_chunks", final.get("relevant_chunks", 0)),
            "total_chunks": meta.get("total_chunks", final.get("total_chunks", 0)),
            "attempts": meta.get("attempts", final.get("attempts", 1)),
            "rewrite_happened": meta.get("rewrite_happened", False),
            "rewritten_question": meta.get("rewritten_question"),
            "web_used": final.get("web_used", False),
            "augmented": final.get("augmented", False),
            "refined": final.get("refined", False),
            "verified": final.get("verified", True),
            "regenerated": final.get("regenerated", False),
            "unsupported_claims": final.get("unsupported_claims", []),
            "sources": final.get("sources", []),
        }

        # 3. Badges, refined query & citations (answer already streamed above)
        _render_assistant_message(msg, show_content=False)

    # 4. Save assistant message in state
    st.session_state.messages.append(msg)