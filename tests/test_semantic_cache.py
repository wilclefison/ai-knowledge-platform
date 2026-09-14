import pytest
from src.cache.semantic_cache import SemanticCacheEngine

def test_semantic_cache_hit_and_financial_savings():
    cache = SemanticCacheEngine(similarity_threshold=0.88)
    tenant = "tenant_test_101"
    
    # Store initial answer
    cache.store(
        tenant_id=tenant,
        query="What are the IAM requirements for security in 2024?",
        answer="All administrative roles must enforce MFA.",
        citations=[{"chunk_id": "c1", "title": "IAM Whitepaper"}],
        tokens_saved=1500
    )
    
    # Query with exact or slight variation
    result = cache.lookup(tenant, "What are the IAM requirements for security in 2024?")
    
    assert result.is_hit is True
    assert result.similarity_score >= 0.88
    assert result.cached_answer == "All administrative roles must enforce MFA."
    assert result.tokens_saved == 1500
    assert result.latency_ms < 10.0

def test_semantic_cache_isolates_tenants():
    cache = SemanticCacheEngine()
    
    cache.store(
        tenant_id="tenant_A",
        query="How to configure VPN?",
        answer="VPN config for Tenant A.",
        citations=[],
        tokens_saved=500
    )
    
    # Tenant B queries the same question -> MUST BE A MISS
    result_b = cache.lookup("tenant_B", "How to configure VPN?")
    assert result_b.is_hit is False
