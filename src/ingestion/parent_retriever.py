from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import uuid
import time
import logging

logger = logging.getLogger(__name__)

class ChildChunk(BaseModel):
    child_id: str
    parent_id: str
    content: str
    token_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ParentDocument(BaseModel):
    parent_id: str
    tenant_id: str
    title: str
    content: str
    token_count: int
    child_chunks: List[ChildChunk] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class HierarchicalSearchResult(BaseModel):
    parent_id: str
    matched_child_id: str
    parent_title: str
    parent_content: str
    child_matched_content: str
    similarity_score: float
    retrieval_strategy: str = "Small-to-Big (Parent-Document)"

class ParentDocumentRetriever:
    """
    Production-grade Parent Document & Hierarchical Summary Retriever.
    Decouples retrieval granularity (small child chunks) from generation context (large parent docs).
    """
    def __init__(self, parent_chunk_size: int = 800, child_chunk_size: int = 150):
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self._parent_store: Dict[str, ParentDocument] = {}
        self._child_store: Dict[str, ChildChunk] = {}

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text.strip()) // 4)

    def split_and_index(
        self,
        tenant_id: str,
        title: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ParentDocument:
        """
        Splits large document into rich parent sections, then creates micro child chunks for vector indexing.
        """
        parent_id = f"parent_{uuid.uuid4().hex[:8]}"
        meta = metadata or {}
        
        # In a full pipeline, parent can be split by section headers (# H1, ## H2)
        parent_tokens = self._estimate_tokens(text)
        parent_doc = ParentDocument(
            parent_id=parent_id,
            tenant_id=tenant_id,
            title=title,
            content=text,
            token_count=parent_tokens,
            metadata=meta
        )

        # Generate child micro-chunks (~150 tokens) from the parent text
        words = text.split()
        words_per_child = self.child_chunk_size
        
        for i in range(0, len(words), words_per_child):
            child_text = " ".join(words[i:i + words_per_child])
            if len(child_text.strip()) < 10:
                continue
                
            child_id = f"child_{uuid.uuid4().hex[:8]}"
            child = ChildChunk(
                child_id=child_id,
                parent_id=parent_id,
                content=child_text,
                token_count=self._estimate_tokens(child_text),
                metadata=meta
            )
            parent_doc.child_chunks.append(child)
            self._child_store[child_id] = child

        self._parent_store[parent_id] = parent_doc
        logger.info(f"Indexed Parent {parent_id} with {len(parent_doc.child_chunks)} micro-chunks")
        return parent_doc

    def resolve_parent_from_child(
        self,
        child_id: str,
        score: float
    ) -> Optional[HierarchicalSearchResult]:
        """
        Given a matched child chunk ID from vector search, resolves and returns the complete parent document.
        """
        child = self._child_store.get(child_id)
        if not child:
            return None

        parent = self._parent_store.get(child.parent_id)
        if not parent:
            return None

        return HierarchicalSearchResult(
            parent_id=parent.parent_id,
            matched_child_id=child.child_id,
            parent_title=parent.title,
            parent_content=parent.content,
            child_matched_content=child.content,
            similarity_score=round(score, 4)
        )

parent_retriever = ParentDocumentRetriever()
