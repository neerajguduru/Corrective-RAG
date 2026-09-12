import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Corrective RAG",
    layout="wide"
)

st.title("📚 Corrective RAG Assistant")

st.divider()

uploaded = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded:

    files = {
        "file": (
            uploaded.name,
            uploaded,
            "application/pdf",
        )
    }

    response = requests.post(
        f"{API_URL}/upload",
        files=files,
    )

    st.success(response.json()["message"])

st.divider()

question = st.text_input(
    "Ask a Question"
)

if st.button("Generate Answer"):

    if question.strip():

        with st.spinner("Thinking..."):

            response = requests.post(
                f"{API_URL}/chat",
                params={
                    "question": question
                },
            )

            data = response.json()

            grade = data.get("grade")
            rewrite_happened = data.get("rewrite_happened", False)
            rewritten_question = data.get("rewritten_question")
            answer = data["answer"]

            st.divider()

            col1, col2 = st.columns(2)

            with col1:
                if grade == "YES":
                    st.success("✅ Grade: YES — Docs were relevant, answered directly")
                else:
                    st.warning("⚠️ Grade: NO — Docs were not relevant")

            with col2:
                if rewrite_happened:
                    st.info(f"🔄 Query was rewritten")
                else:
                    st.info("➡️ No rewrite needed")

            if rewrite_happened and rewritten_question:
                st.caption(f"**Rewritten question:** {rewritten_question}")

            st.markdown("### Answer")
            st.write(answer)