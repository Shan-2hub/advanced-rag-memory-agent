
from langchain_core.documents import Document


def reciprocal_rank_fusion(
    dense_results: list[Document],
    sparse_results: list[Document],
    k: int = 60,
    top_n: int = 20,
) -> list[Document]:

    scores = {}
    documents = {}

    result_lists = [
        dense_results,
        sparse_results,
    ]

    for result_list in result_lists:

        for rank, document in enumerate(
            result_list,
            start=1
        ):

            child_id = document.metadata.get("child_id")

            if not child_id:
                raise ValueError(
                    "Retrieved document is missing child_id."
                )

            scores[child_id] = (
                scores.get(child_id, 0)
                + 1 / (k + rank)
            )

            documents[child_id] = document

    ranked_documents = sorted(
        documents.values(),
        key=lambda doc: scores[
            doc.metadata["child_id"]
        ],
        reverse=True,
    )

    return ranked_documents[:top_n]
