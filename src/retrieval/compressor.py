from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import time
import re
import logging

logger = logging.getLogger(__name__)

class CompressedChunk(BaseModel):
    chunk_id: str
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float = Field(..., description="e.g. 0.52 = 48% reduction in tokens")
    retained_sentences_count: int
    total_sentences_count: int

class ContextualCompressor:
    """
    Sentence-level Contextual Compression Engine (LLMLingua pattern).
    Dynamically prunes low-information filler sentences, redundant boilerplate,
    and noise from retrieved chunks while preserving key facts and numbers.
    """
    def __init__(self, relevance_threshold: float = 0.25):
        self.relevance_threshold = relevance_threshold

    def _estimate_tokens(self, text: str) -> int:
        """Heuristic token estimator (~4 characters per token for English)."""
        return max(1, len(text.strip()) // 4)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Splits document chunk into individual semantic sentences."""
        # Handles periods, question marks, and newlines
        sentences = re.split(r'(?<=[.!?\n])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 3]

    def compress_chunk(self, query: str, chunk_id: str, content: str) -> CompressedChunk:
        """
        Compresses a single chunk by scoring and filtering individual sentences.
        """
        sentences = self._split_into_sentences(content)
        if not sentences:
            tokens = self._estimate_tokens(content)
            return CompressedChunk(
                chunk_id=chunk_id,
                original_text=content,
                compressed_text=content,
                original_tokens=tokens,
                compressed_tokens=tokens,
                compression_ratio=1.0,
                retained_sentences_count=0,
                total_sentences_count=0
            )

        query_terms = set(re.findall(r'\w+', query.lower()))
        retained_sentences = []

        for sentence in sentences:
            sentence_words = set(re.findall(r'\w+', sentence.lower()))
            
            # Check for direct keyword overlap
            overlap = len(query_terms.intersection(sentence_words))
            overlap_score = overlap / max(1, len(query_terms))
            
            # Boost sentences containing numeric data, policies, or technical specifications
            has_numbers = bool(re.search(r'\d+', sentence))
            has_specs = any(keyword in sentence.lower() for keyword in ["must", "require", "mandatory", "shall", "policy", "rule", "standard", "mfa"])
            
            boost = 0.2 if has_numbers or has_specs else 0.0
            total_relevance = overlap_score + boost

            # Keep sentences that exceed the relevance threshold or contain critical directives
            if total_relevance >= self.relevance_threshold or has_specs:
                retained_sentences.append(sentence)

        # Fallback: If all sentences were filtered, retain the first sentence
        if not retained_sentences and sentences:
            retained_sentences.append(sentences[0])

        compressed_text = " ".join(retained_sentences)
        orig_tokens = self._estimate_tokens(content)
        comp_tokens = self._estimate_tokens(compressed_text)
        comp_ratio = round(comp_tokens / max(1, orig_tokens), 3)

        return CompressedChunk(
            chunk_id=chunk_id,
            original_text=content,
            compressed_text=compressed_text,
            original_tokens=orig_tokens,
            compressed_tokens=comp_tokens,
            compression_ratio=comp_ratio,
            retained_sentences_count=len(retained_sentences),
            total_sentences_count=len(sentences)
        )

    def compress_candidates(
        self,
        query: str,
        candidates: List[Dict[str, Any]]
    ) -> List[CompressedChunk]:
        """Compresses a batch of retrieved/reranked candidate chunks."""
        results = []
        for c in candidates:
            chunk_id = str(c.get("id") or c.get("chunk_id", "chunk_unknown"))
            content = c.get("content", "")
            compressed = self.compress_chunk(query, chunk_id, content)
            results.append(compressed)
        return results

compressor = ContextualCompressor()
