
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def split_documents(
    documents: list[Document],
    parent_chunk_size: int = 1000,
    child_chunk_size: int = 200,
):
    """
    Split documents into parent and child chunks.

    Each child stores the ID of its parent.
    """

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=30,
        separators=["\n", ". ", " ", ""],
    )

    parent_documents = []
    child_documents = []

    for document in documents:

        parents = parent_splitter.split_documents(
            [document]
        )

        for parent in parents:

            parent_id = str(uuid.uuid4())

            parent.metadata["parent_id"] = parent_id

            parent_documents.append(
                Document(
                    page_content=parent.page_content,
                    metadata=parent.metadata.copy(),
                )
            )

            children = child_splitter.split_documents(
                [parent]
            )

            for child in children:

                child_metadata = child.metadata.copy()

                child_metadata["parent_id"] = parent_id

                child_metadata["child_id"] = str(uuid.uuid4())
                
                child_metadata["source"] = (
                    document.metadata.get("source", "unknown")
                )

                child_documents.append(
                    Document(
                        page_content=child.page_content,
                        metadata=child_metadata,
                    )
                )

    return parent_documents, child_documents
