
import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from app.ingestion.storage import load_saved_documents

load_dotenv()


def main():

    children = load_saved_documents(
        "data/children.jsonl"
    )

    if not children:
        print("No child documents found.")
        return

    child_ids = [
        doc.metadata["child_id"]
        for doc in children
    ]

    embeddings = OpenAIEmbeddings(
        model=os.getenv(
            "EMBEDDING_MODEL",
            "text-embedding-3-small"
        )
    )

    QdrantVectorStore.from_documents(
        documents=children,
        embedding=embeddings,
        ids=child_ids,
        url=os.getenv(
            "QDRANT_URL",
            "http://localhost:6333"
        ),
        collection_name=os.getenv(
            "QDRANT_COLLECTION",
            "advanced_rag"
        ),
        force_recreate=True,
    )

    print(f"Successfully indexed {len(children)} child chunks.")


if __name__ == "__main__":
    main()
