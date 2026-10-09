import pytest
from src.retrieval.self_rag import SelfRAGEngine

def test_self_rag_supported_generation():
    engine = SelfRAGEngine()
    context = [{"content": "We grew by 20% in Q3."}]
    
    result = engine.generate_with_reflection("What was the Q3 growth?", context)
    
    assert result.critique.is_supported is True
    assert result.critique.needs_rewrite is False
    assert result.original_query == "What was the Q3 growth?"
    assert result.latency_ms > 0

def test_self_rag_hallucination_detection():
    engine = SelfRAGEngine()
    # Empty context, but asking for metrics - should trigger hallucination rewrite
    context = []
    
    result = engine.generate_with_reflection("What are the Q3 financial metrics?", context)
    
    assert result.critique.is_supported is False
    assert result.critique.needs_rewrite is True
    assert "cannot answer" in result.final_response.lower()
    assert "[IS_SUP]" in result.critique.critique_reason
