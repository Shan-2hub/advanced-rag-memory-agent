
import os

from dotenv import load_dotenv
from langchain_cohere import CohereRerank
from langchain_core.documents import Document

load_dotenv()


class CohereDocumentReranker:

    def __init__(self):

        self.top_n = int(
            os.getenv("RERANK_TOP_N", "5")
        )

        self.min_score = float(
            os.getenv("RERANK_MIN_SCORE", "0.0")
        )

        self.reranker = CohereRerank(
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            model=os.getenv(
                "RERANK_MODEL",
                "rerank-english-v3.0"
            ),
            top_n=self.top_n,
        )

    def rerank(
        self,
        query: str,
        documents: list[Document],
    ) -> list[Document]:

        if not documents:
            return []

        # Ask Cohere to rank the candidate documents
        ranked_documents = self.reranker.compress_documents(
            documents,
            query,
        )

        # Filter documents using relevance scores
        filtered_documents = [
            document
            for document in ranked_documents
            if float(
                document.metadata.get("relevance_score", 0.0)
            ) >= self.min_score
        ]

        return filtered_documents[:self.top_n]
