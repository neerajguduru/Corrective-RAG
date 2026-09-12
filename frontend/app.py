import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

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

    # 2. Upload Section
    st.subheader("⬆️ Upload PDF")
    uploaded = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload new documents to chunk and embed into ChromaDB",
        key="pdf_uploader",
    )

    if uploaded:
        if st.button("Index Document", type="secondary", use_container_width=True):
            with st.spinner(f"Ingesting & embedding {uploaded.name}..."):
                files = {"file": (uploaded.name, uploaded, "application/pdf")}
                try:
                    response = requests.post(f"{API_URL}/upload", files=files)
                    if response.ok:
                        st.toast(f"Indexed {uploaded.name} successfully!", icon="✅")
                        st.rerun()
                    else:
                        st.error(f"Upload failed: {response.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

    st.divider()

    # 3. Actions & Status
    st.button("🗑️ Clear Chat History", on_click=clear_chat, use_container_width=True)

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
            grade = msg.get("grade")
            rewrite_happened = msg.get("rewrite_happened", False)
            rewritten_q = msg.get("rewritten_question")
            sources = msg.get("sources", [])

            cols = st.columns([1, 1, 3])
            with cols[0]:
                if grade == "YES":
                    st.markdown("**:green[● Grade: Relevant]**")
                else:
                    st.markdown("**:orange[● Grade: Insufficient]**")
            with cols[1]:
                if rewrite_happened:
                    st.markdown("**:blue[🔄 Query Rewritten]**")
                else:
                    st.markdown("**:gray[➡️ Direct Retrieval]**")

            if rewrite_happened and rewritten_q:
                st.caption(f"**Refined search query:** *{rewritten_q}*")

            st.markdown(msg["content"])

            if sources:
                with st.expander(f"📑 View {len(sources)} Source Citations", expanded=False):
                    for idx, src in enumerate(sources, 1):
                        with st.container(border=True):
                            st.markdown(f"**{idx}. {src['doc']}** · *Page {src['page']}*")
                            st.caption(src["snippet"] + ("..." if len(src["snippet"]) >= 300 else ""))

# ── Chat Input ────────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask a question about your documents...")

if prompt:
    # 1. Display and record user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 2. Assistant response
    with st.chat_message("assistant"):
        with st.status("Evaluating retrieval & generating answer...", expanded=True) as status_box:
            try:
                st.write("🔍 Retrieving chunks from vector database...")
                response = requests.post(
                    f"{API_URL}/chat",
                    params={
                        "question": prompt,
                        "selected_docs": selected_docs,
                    },
                    timeout=120,
                )
                data = response.json()

                grade = data.get("grade")
                rewrite_happened = data.get("rewrite_happened", False)
                rewritten_q = data.get("rewritten_question")
                sources = data.get("sources", [])
                answer = data.get("answer", "")

                if rewrite_happened:
                    st.write(f"🔄 Initial chunks lacked relevance. Rewritten query to: *{rewritten_q}*")
                else:
                    st.write("✅ Retrieved context evaluated as sufficient.")

                status_box.update(label="Complete!", state="complete", expanded=False)

            except Exception as e:
                status_box.update(label="Error occurred", state="error", expanded=True)
                st.error(f"Failed to generate answer: {e}")
                st.stop()

        # Display badges & answer
        cols = st.columns([1, 1, 3])
        with cols[0]:
            if grade == "YES":
                st.markdown("**:green[● Grade: Relevant]**")
            else:
                st.markdown("**:orange[● Grade: Insufficient]**")
        with cols[1]:
            if rewrite_happened:
                st.markdown("**:blue[🔄 Query Rewritten]**")
            else:
                st.markdown("**:gray[➡️ Direct Retrieval]**")

        if rewrite_happened and rewritten_q:
            st.caption(f"**Refined search query:** *{rewritten_q}*")

        st.markdown(answer)

        if sources:
            with st.expander(f"📑 View {len(sources)} Source Citations", expanded=False):
                for idx, src in enumerate(sources, 1):
                    with st.container(border=True):
                        st.markdown(f"**{idx}. {src['doc']}** · *Page {src['page']}*")
                        st.caption(src["snippet"] + ("..." if len(src["snippet"]) >= 300 else ""))

    # 3. Save assistant message in state
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "grade": grade,
        "rewrite_happened": rewrite_happened,
        "rewritten_question": rewritten_q,
        "sources": sources,
    })