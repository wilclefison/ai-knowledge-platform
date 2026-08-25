import pytest
from src.ingestion.chunker import SemanticChunker

def test_semantic_chunker_empty_input():
    chunker = SemanticChunker(target_chunk_size=500, overlap=50)
    assert chunker.split_text("") == []
    assert chunker.split_text("   ") == []

def test_semantic_chunker_splits_paragraphs():
    chunker = SemanticChunker(target_chunk_size=100, overlap=20)
    text = (
        "# Section 1\n\n"
        "This is a long paragraph detailing the security architecture of the cloud system. "
        "It contains multiple sentences that provide context.\n\n"
        "# Section 2\n\n"
        "This is another distinct section focusing on database replication and pgvector indexing."
    )
    chunks = chunker.split_text(text, page_number=2)
    assert len(chunks) >= 2
    assert all(c.page_number == 2 for c in chunks)
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert "Section 1" in chunks[0].content or "security architecture" in chunks[0].content
