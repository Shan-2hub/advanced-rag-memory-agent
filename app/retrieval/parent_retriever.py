
from app.ingestion.storage import load_saved_documents


class ParentDocumentRetriever:

    def __init__(self):

        parents = load_saved_documents(
            "data/parents.jsonl"
        )

        self.parent_store = {
            doc.metadata["parent_id"]: doc
            for doc in parents
        }

    def retrieve_parents(self, child_documents):

        retrieved_parents = []
        seen_parent_ids = set()

        for child in child_documents:

            parent_id = child.metadata.get("parent_id")

            if not parent_id:
                continue

            if parent_id in seen_parent_ids:
                continue

            parent = self.parent_store.get(parent_id)

            if parent is not None:
                retrieved_parents.append(parent)
                seen_parent_ids.add(parent_id)

        return retrieved_parents
