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
    ) -> list[Document]:

        return self.retriever.invoke(question)