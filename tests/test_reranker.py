
from unittest.mock import MagicMock

from langchain_core.documents import Document

from app.retrieval.reranker import CohereDocumentReranker


def create_document(content, score):
    return Document(
        page_content=content,
        metadata={"relevance_score": score},
    )


def test_reranker_filters_low_scores():

    reranker = CohereDocumentReranker.__new__(
        CohereDocumentReranker
    )

    reranker.top_n = 5
    reranker.min_score = 0.5

    reranker.reranker = MagicMock()

    mock_results = [
        create_document("Relevant document A", 0.95),
        create_document("Relevant document B", 0.85),
        create_document("Weak document", 0.25),
    ]

    reranker.reranker.compress_documents.return_value = (
        mock_results
    )

    results = reranker.rerank(
        "Explain hybrid retrieval",
        mock_results,
    )

    assert len(results) == 2

    assert all(
        doc.metadata["relevance_score"] >= 0.5
        for doc in results
    )


def test_reranker_returns_top_five():

    reranker = CohereDocumentReranker.__new__(
        CohereDocumentReranker
    )

    reranker.top_n = 5
    reranker.min_score = 0.0

    reranker.reranker = MagicMock()

    mock_results = [
        create_document(f"Document {i}", 0.9 - i * 0.05)
        for i in range(10)
    ]

    reranker.reranker.compress_documents.return_value = (
        mock_results[:5]
    )

    results = reranker.rerank(
        "Explain RAG",
        mock_results,
    )

    assert len(results) == 5


def test_reranker_handles_empty_documents():

    reranker = CohereDocumentReranker.__new__(
        CohereDocumentReranker
    )

    reranker.top_n = 5
    reranker.min_score = 0.0
    reranker.reranker = MagicMock()

    results = reranker.rerank("Test query", [])

    assert results == []

    reranker.reranker.compress_documents.assert_not_called()
