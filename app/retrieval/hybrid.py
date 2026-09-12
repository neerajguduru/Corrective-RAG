import re

from langchain_core.documents import Document

from app.config import RERANK_ENABLED, RERANK_MODEL, RERANK_TOP_N


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    k: int = 60,
    top_n: int = 10,
) -> list[Document]:
    """Fuse multiple ranked result lists into one via Reciprocal Rank Fusion.

    RRF score(doc) = sum over lists of 1 / (k + rank). Chunks are deduped
    by page_content so a chunk found by both retrievers gets boosted.
    """

    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            docs.setdefault(key, doc)

    ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)

    return [docs[key] for key in ranked_keys[:top_n]]


class HybridRetriever:
    """Dense (Chroma) + sparse (BM25) retrieval fused with RRF.

    The BM25 corpus is rebuilt lazily: whenever the vector store's chunk
    count changes, all chunks (with embeddings already computed) are
    pulled once and tokenized. This stays cheap at local-corpus scale.
    """

    def __init__(self, vector_db):

        self._vector_db = vector_db
        self._corpus: list[Document] = []
        self._bm25 = None
        self._rebuild_if_stale()

    def _rebuild_if_stale(self):

        try:
            count = self._vector_db._collection.count()
        except Exception:
            count = 0

        if count == len(self._corpus):
            return

        if count == 0:
            self._corpus = []
            self._bm25 = None
            return

        result = self._vector_db._collection.get(include=["documents", "metadatas"])
        self._corpus = [
            Document(page_content=text or "", metadata=meta or {})
            for text, meta in zip(result["documents"], result["metadatas"])
        ]
        self._bm25 = _build_bm25(self._corpus)

    def retrieve(self, question: str, top_n: int, doc_filter: list[str] | None = None) -> list[Document]:
        """Hybrid retrieve: dense + BM25 → RRF fusion → top_n candidates."""

        self._rebuild_if_stale()

        # 1. Dense retrieval (with optional metadata filter)
        if doc_filter:
            dense = self._vector_db.similarity_search(
                question,
                k=top_n,
                filter={"doc_name": {"$in": doc_filter}},
            )
        else:
            dense = self._vector_db.similarity_search(question, k=top_n)

        # 2. BM25 retrieval (filtered in Python — local corpora are small)
        sparse: list[Document] = []
        if self._bm25 is not None:
            scores = self._bm25.get_scores(_tokenize(question))
            ranked = sorted(
                enumerate(scores), key=lambda pair: pair[1], reverse=True
            )
            for idx, score in ranked:
                if score <= 0:
                    break
                doc = self._corpus[idx]
                if doc_filter and doc.metadata.get("doc_name") not in doc_filter:
                    continue
                sparse.append(doc)
                if len(sparse) >= top_n:
                    break

        if not sparse:
            return dense

        # 3. Fuse
        return reciprocal_rank_fusion([dense, sparse], top_n=top_n)


def _build_bm25(corpus: list[Document]):

    from rank_bm25 import BM25Okapi

    return BM25Okapi([_tokenize(doc.page_content) for doc in corpus])


class Reranker:
    """Cross-encoder reranker (BAAI/bge-reranker-base by default).

    Scores (query, chunk) pairs jointly — far more precise than the
    bi-encoder similarity used for retrieval. Loaded lazily so app
    startup and tests don't pay the model cost unless reranking fires.
    """

    def __init__(self):

        self._model = None

    def _ensure_model(self):

        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(RERANK_MODEL, max_length=512)
        return self._model

    def rerank(self, question: str, docs: list[Document]) -> list[Document]:
        """Return docs re-sorted by cross-encoder relevance, truncated to
        RERANK_TOP_N."""

        if not RERANK_ENABLED or len(docs) <= 1:
            return docs[:RERANK_TOP_N]

        model = self._ensure_model()

        pairs = [(question, doc.page_content) for doc in docs]
        scores = model.predict(pairs)

        ranked = sorted(
            zip(docs, scores), key=lambda pair: pair[1], reverse=True
        )

        return [doc for doc, _ in ranked[:RERANK_TOP_N]]
