import pytest
from src.retrieval.reranker import CrossEncoderReranker

def test_reranker_boosts_relevant_chunk():
    reranker = CrossEncoderReranker()
    query = "ISO 27001 compliance audit"
    
    candidates = [
        {"id": "c1", "content": "General employee onboarding checklist and handbook.", "score": 0.03},
        {"id": "c2", "content": "Billing information and payment invoice instructions.", "score": 0.02},
        {"id": "c3", "content": "Mandatory ISO 27001 compliance audit requirements and security review timeline.", "score": 0.01}
    ]
    
    # In the candidate pool, c3 was in 3rd place with low initial score.
    # The Cross-Encoder should identify c3 as the most relevant and promote it to Rank #1!
    results = reranker.rerank(query, candidates, top_k=2)
    
    assert len(results) == 2
    assert results[0].chunk_id == "c3"  # Promoted to Rank #1!
    assert results[0].new_rank == 1
    assert results[0].original_rank == 3
    assert results[0].rerank_score > results[1].rerank_score

def test_reranker_handles_empty_candidates():
    reranker = CrossEncoderReranker()
    assert reranker.rerank("any query", []) == []
