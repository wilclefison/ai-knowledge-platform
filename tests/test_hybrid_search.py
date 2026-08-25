import pytest
from src.retrieval.hybrid_search import HybridSearchEngine

def test_rrf_scoring_favors_dual_matches():
    engine = HybridSearchEngine(rrf_k=60)
    
    dense_results = [
        {"id": "chunk_A", "document_id": "doc_1", "content": "Dense rank 1", "page_number": 1},
        {"id": "chunk_B", "document_id": "doc_1", "content": "Dense rank 2", "page_number": 2},
    ]
    sparse_results = [
        {"id": "chunk_B", "document_id": "doc_1", "content": "Dense rank 2", "page_number": 2},
        {"id": "chunk_C", "document_id": "doc_2", "content": "Sparse rank 2", "page_number": 3},
    ]
    
    # chunk_B appears in BOTH dense (rank 2) and sparse (rank 1), so its cumulative RRF score should be highest!
    results = engine.reciprocal_rank_fusion(dense_results, sparse_results, top_k=3)
    
    assert len(results) == 3
    assert results[0].chunk_id == "chunk_B"  # The dual match wins!
    assert results[0].dense_rank == 2
    assert results[0].sparse_rank == 1
    assert results[0].score > results[1].score
