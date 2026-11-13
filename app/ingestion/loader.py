
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_documents(directory: str) -> list[Document]:
    """
    Load PDF, TXT and Markdown documents
    from the specified directory.
    """
    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )

    documents = []

    for file_path in directory_path.rglob("*"):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
            else:
                loader = TextLoader(
                    str(file_path),
                    encoding="utf-8"
                )

            loaded_docs = loader.load()

            for doc in loaded_docs:
                doc.metadata["source"] = str(file_path)
                documents.append(doc)

        except Exception as error:
            print(f"Failed to load {file_path}: {error}")

    return documents
