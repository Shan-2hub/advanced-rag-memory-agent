
import json
from pathlib import Path

from langchain_core.documents import Document


def save_documents(
    documents: list[Document],
    output_path: str,
):
    """Persist documents as JSONL."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:

        for document in documents:

            record = {
                "page_content": document.page_content,
                "metadata": document.metadata,
            }

            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def load_saved_documents(
    input_path: str,
) -> list[Document]:
    """Load persisted documents from JSONL."""

    path = Path(input_path)

    if not path.exists():
        return []

    documents = []

    with path.open("r", encoding="utf-8") as file:

        for line in file:

            record = json.loads(line)

            documents.append(
                Document(
                    page_content=record["page_content"],
                    metadata=record["metadata"],
                )
            )

    return documents
