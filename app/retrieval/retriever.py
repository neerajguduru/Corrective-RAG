from pathlib import Path

from langchain_core.documents import Document

from app.config import TOP_K
from app.retrieval.vectordb import vector_db


class Retriever:

    def __init__(self):

        self.retriever = vector_db.as_retriever(
            search_kwargs={
                "k": TOP_K
            }
        )

    def retrieve(
        self,
        question: str,
        doc_filter: list[str] | None = None,
    ) -> list[Document]:

        if doc_filter:
            # Filter to specific documents by name
            return vector_db.similarity_search(
                question,
                k=TOP_K,
                filter={"doc_name": {"$in": doc_filter}},
            )

        return self.retriever.invoke(question)

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