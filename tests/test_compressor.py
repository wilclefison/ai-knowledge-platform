import pytest
from src.retrieval.compressor import ContextualCompressor

def test_compressor_prunes_filler_sentences():
    comp = ContextualCompressor(relevance_threshold=0.3)
    query = "What are the data retention requirements?"
    
    content = (
        "Welcome to our company knowledge base. "
        "This document was created in 2021 by the IT committee. "
        "All customer audit logs must be retained for a mandatory minimum of 365 days. "
        "Please contact support if you have general questions about email settings."
    )
    
    result = comp.compress_chunk(query, "chunk_1", content)
    
    # Assert filler sentences were stripped while the 365 days retention policy is preserved
    assert "365 days" in result.compressed_text
    assert "Welcome to our company" not in result.compressed_text
    assert result.compressed_tokens < result.original_tokens
    assert result.compression_ratio < 0.60
    assert result.retained_sentences_count == 1
    assert result.total_sentences_count == 4

def test_compressor_handles_empty_chunk():
    comp = ContextualCompressor()
    result = comp.compress_chunk("query", "c_empty", "")
    assert result.compressed_text == ""
