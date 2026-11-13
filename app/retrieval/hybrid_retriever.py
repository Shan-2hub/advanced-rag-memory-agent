
import time

from app.retrieval.dense_search import DenseRetriever
from app.retrieval.sparse_search import SparseRetriever
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.parent_retriever import (
    ParentDocumentRetriever,
)
from app.retrieval.reranker import (
    CohereDocumentReranker,
)


class HybridRetriever:

    def __init__(self):

        self.dense = DenseRetriever()
        self.sparse = SparseRetriever()
        self.parent = ParentDocumentRetriever()
        self.reranker = CohereDocumentReranker()

    def retrieve(self, query: str):

        retrieval_start = time.perf_counter()

        dense_results = self.dense.search(query, k=20)

        sparse_results = self.sparse.search(query, k=20)

        fused_results = reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            k=60,
            top_n=20,
        )

        parent_documents = self.parent.retrieve_parents(
            fused_results
        )

        retrieval_latency = (
            time.perf_counter() - retrieval_start
        ) * 1000

        rerank_start = time.perf_counter()

        reranked_documents = self.reranker.rerank(
            query,
            parent_documents,
        )

        rerank_latency = (
            time.perf_counter() - rerank_start
        ) * 1000

        return {
            "dense_count": len(dense_results),
            "sparse_count": len(sparse_results),
            "fused_count": len(fused_results),
            "parent_count": len(parent_documents),
            "reranked_count": len(reranked_documents),
            "retrieval_latency_ms": round(
                retrieval_latency, 2
            ),
            "rerank_latency_ms": round(
                rerank_latency, 2
            ),
            "documents": reranked_documents,
        }
