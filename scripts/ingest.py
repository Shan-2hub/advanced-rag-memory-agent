
from app.ingestion.loader import load_documents
from app.ingestion.splitter import split_documents
from app.ingestion.storage import save_documents


DATA_DIRECTORY = "data/documents"


def main():

    print("\nStarting document ingestion...\n")

    # 1. Load original documents
    documents = load_documents(DATA_DIRECTORY)

    if not documents:
        print("No documents found. Add PDF, TXT or MD files.")
        return

    print(f"Loaded documents: {len(documents)}")

    # 2. Generate parent and child chunks
    parents, children = split_documents(documents)

    print(f"Parent chunks: {len(parents)}")
    print(f"Child chunks: {len(children)}")

    # 3. Persist documents
    save_documents(
        parents,
        "data/parents.jsonl"
    )

    save_documents(
        children,
        "data/children.jsonl"
    )

    print("\nIngestion completed successfully!")


if __name__ == "__main__":
    main()
