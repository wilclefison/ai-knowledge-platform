from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from src.config import settings

class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    page_number: int
    score: float
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    retrieval_method: str = "hybrid_rrf"

class HybridSearchEngine:
    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k

    def reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Combines Dense Vector rankings and BM25 Sparse rankings using Reciprocal Rank Fusion (RRF).
        Formula: RRF_Score = sum(1 / (k + rank_i))
        """
        scores: Dict[str, float] = {}
        items: Dict[str, Dict[str, Any]] = {}
        dense_ranks: Dict[str, int] = {}
        sparse_ranks: Dict[str, int] = {}

        # 1. Process Dense Results (Semantic Similarity)
        for rank, item in enumerate(dense_results, start=1):
            chunk_id = str(item["id"])
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))
            items[chunk_id] = item
            dense_ranks[chunk_id] = rank

        # 2. Process Sparse Results (BM25 Keyword Match)
        for rank, item in enumerate(sparse_results, start=1):
            chunk_id = str(item["id"])
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))
            if chunk_id not in items:
                items[chunk_id] = item
            sparse_ranks[chunk_id] = rank

        # 3. Sort by aggregated RRF Score
        sorted_chunk_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)[:top_k]

        results: List[SearchResult] = []
        for cid in sorted_chunk_ids:
            item = items[cid]
            results.append(SearchResult(
                chunk_id=cid,
                document_id=str(item.get("document_id", "")),
                content=item.get("content", ""),
                page_number=item.get("page_number", 1),
                score=round(scores[cid], 5),
                dense_rank=dense_ranks.get(cid),
                sparse_rank=sparse_ranks.get(cid),
                retrieval_method="hybrid_rrf"
            ))

        return results
