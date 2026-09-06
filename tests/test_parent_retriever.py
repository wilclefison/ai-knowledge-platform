import pytest
from src.ingestion.parent_retriever import ParentDocumentRetriever

def test_parent_retriever_indexing_and_resolution():
    retriever = ParentDocumentRetriever(parent_chunk_size=500, child_chunk_size=20)
    
    sample_text = (
        "AWS Cloud Security Architecture Guide. "
        "Section 1 covers IAM Policies. "
        "All IAM administrative roles must enforce mandatory Multi-Factor Authentication. "
        "Section 2 covers Network Peering and security group baseline configs."
    )
    
    parent_doc = retriever.split_and_index(
        tenant_id="tenant_acme",
        title="AWS Security Guide",
        text=sample_text
    )
    
    # Assert parent has child micro-chunks
    assert len(parent_doc.child_chunks) >= 2
    first_child = parent_doc.child_chunks[0]
    
    # Resolve from child ID
    result = retriever.resolve_parent_from_child(first_child.child_id, score=0.95)
    
    assert result is not None
    assert result.parent_id == parent_doc.parent_id
    assert result.parent_title == "AWS Security Guide"
    assert result.parent_content == sample_text
    assert result.matched_child_id == first_child.child_id
