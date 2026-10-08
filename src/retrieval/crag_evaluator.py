import time
import re
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field

class DocumentGrade(str, Enum):
    RELEVANT = "RELEVANT"
    AMBIGUOUS = "AMBIGUOUS"
    IRRELEVANT = "IRRELEVANT"

class CRAGAction(str, Enum):
    CORRECT = "CORRECT"
    AMBIGUOUS_REWRITE = "AMBIGUOUS_REWRITE"
    INCORRECT_FALLBACK = "INCORRECT_FALLBACK"

class GradedChunk(BaseModel):
    chunk_id: str
    original_text: str
    grade: DocumentGrade
    score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    filtered_text: str
    original_tokens: int
    filtered_tokens: int
    tokens_stripped: int

class CRAGReport(BaseModel):
    query: str
    overall_action: CRAGAction
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    total_evaluated: int
    kept_chunks_count: int
    discarded_chunks_count: int
    token_reduction_pct: float
    fallback_query: Optional[str] = None
    graded_chunks: List[GradedChunk]
    execution_latency_ms: float

class CorrectiveRAGEvaluator:
    """
    Autonomous Corrective RAG (CRAG) Document Grader.
    Evaluates retrieved chunks before prompt construction:
    - Filters out noisy or misleading chunks to prevent LLM hallucinations.
    - Strips peripheral filler to maximize prompt Signal-to-Noise Ratio (SNR).
    - Decides corrective actions: CORRECT (proceed), AMBIGUOUS (rewrite/web fallback), or INCORRECT (reject).
    """
    def __init__(self, high_threshold: float = 0.70, low_threshold: float = 0.35):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def grade_documents(self, query: str, chunks: List[Dict[str, Any]]) -> CRAGReport:
        start_time = time.time()
        q_words = set(re.findall(r'\b\w+\b', query.lower()))
        # Remove common stop words for clean token matching
        stopwords = {"the", "a", "an", "in", "on", "at", "for", "to", "of", "and", "is", "are", "what", "how", "why", "qual", "quais", "como", "por", "que", "de", "do", "da", "em", "um", "uma"}
        q_keywords = q_words - stopwords

        graded_chunks: List[GradedChunk] = []
        total_orig_tokens = 0
        total_filt_tokens = 0

        for idx, chunk in enumerate(chunks):
            cid = chunk.get("chunk_id", f"chunk_{idx+1}")
            text = chunk.get("text", chunk.get("content", ""))
            orig_tokens = max(1, len(text.split()))
            total_orig_tokens += orig_tokens

            text_words = set(re.findall(r'\b\w+\b', text.lower()))
            overlap = q_keywords.intersection(text_words)
            overlap_ratio = len(overlap) / max(1, len(q_keywords))

            # Heuristic semantic scoring
            score = round(min(1.0, overlap_ratio * 1.25), 2)

            # Strip irrelevant boilerplate/disclaimer sentences if relevant
            filtered_text = text
            if score >= self.high_threshold:
                grade = DocumentGrade.RELEVANT
                reasoning = f"Direct semantic alignment with query keywords ({', '.join(list(overlap)[:3])})."
                # Strip boilerplates like 'Confidential draft', 'Page X of Y', etc.
                clean_lines = [line for line in text.split('\n') if not re.search(r'^(page \d+|confidential|copyright|all rights reserved)', line.strip().lower())]
                filtered_text = '\n'.join(clean_lines).strip()
            elif score >= self.low_threshold:
                grade = DocumentGrade.AMBIGUOUS
                reasoning = "Partial keyword intersection but lacks complete contextual answer."
                filtered_text = text
            else:
                grade = DocumentGrade.IRRELEVANT
                reasoning = "No significant semantic overlap with query intent. Flagged as noise."
                filtered_text = ""

            filt_tokens = len(filtered_text.split()) if filtered_text else 0
            if grade != DocumentGrade.IRRELEVANT:
                total_filt_tokens += filt_tokens

            stripped = orig_tokens - filt_tokens

            graded_chunks.append(GradedChunk(
                chunk_id=cid,
                original_text=text,
                grade=grade,
                score=score,
                reasoning=reasoning,
                filtered_text=filtered_text,
                original_tokens=orig_tokens,
                filtered_tokens=filt_tokens,
                tokens_stripped=stripped
            ))

        # Determine overall action based on ratio of relevant chunks
        relevant_count = sum(1 for g in graded_chunks if g.grade == DocumentGrade.RELEVANT)
        ambiguous_count = sum(1 for g in graded_chunks if g.grade == DocumentGrade.AMBIGUOUS)
        irrelevant_count = sum(1 for g in graded_chunks if g.grade == DocumentGrade.IRRELEVANT)

        total = len(graded_chunks)
        confidence = round(relevant_count / total, 2) if total > 0 else 0.0

        if confidence >= 0.50 or (relevant_count >= 1 and irrelevant_count == 0):
            action = CRAGAction.CORRECT
            fallback_query = None
        elif (relevant_count + ambiguous_count) > 0:
            action = CRAGAction.AMBIGUOUS_REWRITE
            fallback_query = f"{query} enterprise architecture specifications"
        else:
            action = CRAGAction.INCORRECT_FALLBACK
            fallback_query = f"web_search: {query}"

        reduction_pct = round(((total_orig_tokens - total_filt_tokens) / total_orig_tokens) * 100, 1) if total_orig_tokens > 0 else 0.0
        latency_ms = round((time.time() - start_time) * 1000, 2)

        return CRAGReport(
            query=query,
            overall_action=action,
            confidence_score=confidence,
            total_evaluated=total,
            kept_chunks_count=relevant_count + ambiguous_count,
            discarded_chunks_count=irrelevant_count,
            token_reduction_pct=reduction_pct,
            fallback_query=fallback_query,
            graded_chunks=graded_chunks,
            execution_latency_ms=latency_ms
        )

crag_evaluator = CorrectiveRAGEvaluator()
