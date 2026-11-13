
import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()


class DenseRetriever:

    def __init__(self):

        self.embeddings = OpenAIEmbeddings(
            model=os.getenv(
                "EMBEDDING_MODEL",
                "text-embedding-3-small"
            )
        )

        self.vector_store = (
            QdrantVectorStore.from_existing_collection(
                embedding=self.embeddings,
                collection_name=os.getenv(
                    "QDRANT_COLLECTION",
                    "advanced_rag"
                ),
                url=os.getenv(
                    "QDRANT_URL",
                    "http://localhost:6333"
                ),
            )
        )

    def search(self, query: str, k: int = 20):

        results = self.vector_store.similarity_search(
            query,
            k=k,
        )

        return results
