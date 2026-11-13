
from langchain_core.documents import Document

from app.retrieval.fusion import reciprocal_rank_fusion


def make_doc(child_id, content):
    return Document(
        page_content=content,
        metadata={"child_id": child_id},
    )


def test_rrf_merges_dense_and_sparse_results():

    dense = [
        make_doc("A", "Document A"),
        make_doc("B", "Document B"),
        make_doc("C", "Document C"),
    ]

    sparse = [
        make_doc("B", "Document B"),
        make_doc("A", "Document A"),
        make_doc("D", "Document D"),
    ]

    results = reciprocal_rank_fusion(
        dense,
        sparse,
        k=60,
        top_n=4,
    )

    ids = [
        doc.metadata["child_id"]
        for doc in results
    ]

    assert set(ids) == {"A", "B", "C", "D"}

    # A and B appear in both retrieval lists.
    assert ids.index("A") < ids.index("C")
    assert ids.index("B") < ids.index("D")


def test_rrf_removes_duplicate_documents():

    dense = [
        make_doc("A", "Document A"),
        make_doc("B", "Document B"),
    ]

    sparse = [
        make_doc("A", "Document A"),
        make_doc("B", "Document B"),
    ]

    results = reciprocal_rank_fusion(
        dense,
        sparse,
        top_n=10,
    )

    ids = [
        doc.metadata["child_id"]
        for doc in results
    ]

    assert len(ids) == len(set(ids))
    assert len(results) == 2


def test_rrf_respects_top_n():

    dense = [
        make_doc(str(i), f"Document {i}")
        for i in range(10)
    ]

    results = reciprocal_rank_fusion(
        dense,
        [],
        top_n=3,
    )

    assert len(results) == 3


def test_rrf_empty_results():

    results = reciprocal_rank_fusion([], [])

    assert results == []
