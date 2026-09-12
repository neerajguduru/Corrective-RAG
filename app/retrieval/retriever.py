from pathlib import Path

from langchain_core.documents import Document

from app.config import HYBRID_CANDIDATES
from app.retrieval.hybrid import HybridRetriever, Reranker
from app.retrieval.vectordb import vector_db


class Retriever:

    def __init__(self):

        self.hybrid = HybridRetriever(vector_db)
        self.reranker = Reranker()

    def retrieve(
        self,
        question: str,
        doc_filter: list[str] | None = None,
    ) -> list[Document]:
        """Hybrid retrieve (dense + BM25 → RRF) → cross-encoder rerank → top-K."""

        candidates = self.hybrid.retrieve(
            question,
            top_n=HYBRID_CANDIDATES,
            doc_filter=doc_filter,
        )

        return self.reranker.rerank(question, candidates)

    def get_available_documents(self) -> list[str]:
        """Return list of unique doc names indexed in ChromaDB."""

        try:
            results = vector_db._collection.get(include=["metadatas"])
            doc_names = set()

            for metadata in results["metadatas"]:
                # Use doc_name tag if present, fallback to source path basename
                name = metadata.get("doc_name") or Path(
                    metadata.get("source", "")
                ).name
                if name:
                    doc_names.add(name)

            return sorted(list(doc_names))

        except Exception:
            return []

    def delete_document(self, doc_name: str) -> int:
        """Delete all chunks belonging to doc_name from ChromaDB."""
        ids_to_del = set()
        try:
            # 1. Match by doc_name metadata
            res = vector_db._collection.get(
                where={"doc_name": {"$eq": doc_name}},
                include=[]
            )
            ids_to_del.update(res.get("ids", []))

            # 2. Match by source path filename
            all_res = vector_db._collection.get(include=["metadatas"])
            for chunk_id, meta in zip(all_res["ids"], all_res["metadatas"]):
                if meta:
                    src = meta.get("source", "")
                    if src.endswith(doc_name) or Path(src).name == doc_name:
                        ids_to_del.add(chunk_id)

            if ids_to_del:
                vector_db._collection.delete(ids=list(ids_to_del))
                return len(ids_to_del)
        except Exception as e:
            print(f"Error deleting {doc_name}: {e}")

        return len(ids_to_del)
