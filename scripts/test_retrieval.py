
from app.retrieval.hybrid_retriever import HybridRetriever


def main():

    retriever = HybridRetriever()

    query = "How does semantic retrieval work?"

    print("\nRunning advanced hybrid retrieval...\n")

    results = retriever.retrieve(query)

    print("Dense results:", results["dense_count"])
    print("Sparse results:", results["sparse_count"])
    print("Fused results:", results["fused_count"])
    print("Parent documents:", results["parent_count"])
    print("Reranked documents:", results["reranked_count"])

    print("\nFinal compressed context:\n")

    for index, document in enumerate(
        results["documents"],
        start=1
    ):

        print(f"\nDocument {index}")

        print(
            "Relevance score:",
            document.metadata.get("relevance_score")
        )

        print("-" * 50)
        print(document.page_content[:500])


if __name__ == "__main__":
    main()
