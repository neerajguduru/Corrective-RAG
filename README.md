# Corrective RAG using LangGraph

A Retrieval-Augmented Generation system that evaluates retrieval quality and automatically rewrites user queries when the retrieved documents are insufficient.

---

## Features

- LangGraph Workflow
- Ollama (Llama 3.2)
- ChromaDB
- HuggingFace Embeddings
- PDF Upload
- Automatic PDF Ingestion
- Query Rewriting
- Retrieval Grading
- FastAPI
- Streamlit

---

## Workflow

```
User Question
      │
      ▼
Retrieve Documents
      │
      ▼
Grade Retrieved Documents
      │
  Yes │ No
      ▼
 Generate Answer
      ▲
      │
 Rewrite Query
      │
 Retrieve Again
```

---

## Tech Stack

- LangChain
- LangGraph
- Ollama
- ChromaDB
- FastAPI
- Streamlit

---

## Installation

```bash
pip install -r requirements.txt
```

Run Ollama

```bash
ollama serve
```

Pull model

```bash
ollama pull llama3.2
```

Start API

```bash
uvicorn app.main:app --reload
```

Start UI

```bash
streamlit run frontend/app.py
```

---

## Future Improvements

- Hybrid Retrieval
- Cross Encoder Reranking
- Conversation Memory
- Multi Query Retrieval
- Source Citations
- Evaluation using RAGAS