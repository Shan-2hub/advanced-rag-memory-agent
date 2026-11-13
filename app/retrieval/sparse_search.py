
from langchain_community.retrievers import BM25Retriever
from app.ingestion.storage import load_saved_documents


class SparseRetriever:

    def __init__(self):

        documents = load_saved_documents(
            "data/children.jsonl"
        )

        if not documents:
            raise ValueError(
                "No child documents available for BM25."
            )

        self.retriever = BM25Retriever.from_documents(
            documents
        )

    def search(self, query: str, k: int = 20):

        self.retriever.k = k

        return self.retriever.invoke(query)
