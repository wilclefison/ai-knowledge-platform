import re
from typing import List, Dict, Any
from pydantic import BaseModel

class Chunk(BaseModel):
    chunk_index: int
    content: str
    token_count_approx: int
    page_number: int = 1
    metadata: Dict[str, Any] = {}

class SemanticChunker:
    def __init__(self, target_chunk_size: int = 600, overlap: int = 100):
        self.target_chunk_size = target_chunk_size
        self.overlap = overlap

    def split_text(self, text: str, page_number: int = 1) -> List[Chunk]:
        """
        Splits text while preserving markdown headings and semantic paragraph boundaries.
        """
        if not text or not text.strip():
            return []

        # Split by double newline (paragraphs) or markdown headers
        paragraphs = re.split(r'(\n\n|(?=^#{1,4} ))', text, flags=re.MULTILINE)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks: List[Chunk] = []
        current_chunk_parts: List[str] = []
        current_length = 0
        chunk_idx = 0

        for para in paragraphs:
            para_len = len(para)
            if current_length + para_len > self.target_chunk_size and current_chunk_parts:
                combined_content = " ".join(current_chunk_parts)
                chunks.append(Chunk(
                    chunk_index=chunk_idx,
                    content=combined_content,
                    token_count_approx=len(combined_content.split()),
                    page_number=page_number,
                    metadata={"source_length": len(combined_content)}
                ))
                chunk_idx += 1
                
                # Keep last part for overlap
                if self.overlap > 0 and len(current_chunk_parts) > 1:
                    current_chunk_parts = [current_chunk_parts[-1], para]
                    current_length = len(current_chunk_parts[0]) + para_len
                else:
                    current_chunk_parts = [para]
                    current_length = para_len
            else:
                current_chunk_parts.append(para)
                current_length += para_len

        if current_chunk_parts:
            combined_content = " ".join(current_chunk_parts)
            chunks.append(Chunk(
                chunk_index=chunk_idx,
                content=combined_content,
                token_count_approx=len(combined_content.split()),
                page_number=page_number,
                metadata={"source_length": len(combined_content)}
            ))

        return chunks
