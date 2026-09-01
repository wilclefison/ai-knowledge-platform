from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import time
import math
import logging

logger = logging.getLogger(__name__)

class EvalSample(BaseModel):
    sample_id: str
    query: str
    contexts: List[str]
    generated_answer: str
    ground_truth: str

class MetricResult(BaseModel):
    faithfulness: float = Field(..., ge=0.0, le=1.0, description="1.0 = Zero Hallucination, 100% grounded")
    answer_relevance: float = Field(..., ge=0.0, le=1.0, description="1.0 = Direct answer, no irrelevant fluff")
    context_recall: float = Field(..., ge=0.0, le=1.0, description="1.0 = All ground truth facts retrieved")
    context_precision: float = Field(..., ge=0.0, le=1.0, description="1.0 = High signal chunks ranked at the top")
    passed: bool

class EvalReport(BaseModel):
    total_samples: int
    mean_faithfulness: float
    mean_answer_relevance: float
    mean_context_recall: float
    mean_context_precision: float
    overall_rag_score: float
    samples: List[Dict[str, Any]]
    execution_time_ms: float
    status: str = "PASSED"

class RagasEvaluator:
    """
    Automated Continuous Evaluation Engine for Enterprise RAG.
    Computes mathematical metrics for Faithfulness, Answer Relevance, Context Recall, and Context Precision.
    """
    def __init__(self, pass_threshold: float = 0.85):
        self.pass_threshold = pass_threshold

    def compute_faithfulness(self, contexts: List[str], generated_answer: str) -> float:
        """
        Measures the proportion of claims in the generated answer that are directly supported by the context.
        Score = Supported Claims / Total Claims
        """
        combined_context = " ".join(contexts).lower()
        sentences = [s.strip() for s in generated_answer.split(".") if len(s.strip()) > 5]
        if not sentences:
            return 1.0

        supported_claims = 0
        for s in sentences:
            s_words = set(s.lower().split())
            matched_words = sum(1 for w in s_words if w in combined_context)
            if (matched_words / max(1, len(s_words))) >= 0.6:
                supported_claims += 1

        return round(supported_claims / len(sentences), 4)

    def compute_answer_relevance(self, query: str, generated_answer: str) -> float:
        """
        Measures if the generated answer directly addresses the intent of the query.
        """
        query_words = set(query.lower().split())
        answer_words = set(generated_answer.lower().split())
        
        intersection = query_words.intersection(answer_words)
        relevance = len(intersection) / max(1, len(query_words))
        
        # Sigmoid curve normalization
        normalized = 1.0 / (1.0 + math.exp(-relevance * 6 + 3))
        return round(min(1.0, max(0.0, normalized)), 4)

    def compute_context_recall(self, contexts: List[str], ground_truth: str) -> float:
        """
        Measures if all key facts from the ground truth exist in the retrieved context.
        """
        combined_context = " ".join(contexts).lower()
        gt_words = set(ground_truth.lower().split())
        
        matches = sum(1 for w in gt_words if w in combined_context)
        recall = matches / max(1, len(gt_words))
        return round(min(1.0, recall), 4)

    def compute_context_precision(self, contexts: List[str], ground_truth: str) -> float:
        """
        Calculates precision weighted by the rank order of relevant contexts.
        """
        if not contexts:
            return 0.0

        gt_words = set(ground_truth.lower().split())
        precisions = []
        relevant_count = 0

        for rank, ctx in enumerate(contexts, start=1):
            ctx_words = set(ctx.lower().split())
            overlap = len(gt_words.intersection(ctx_words))
            if overlap >= 2:
                relevant_count += 1
                precisions.append(relevant_count / rank)

        if not precisions:
            return 0.0
        return round(sum(precisions) / len(precisions), 4)

    def evaluate_sample(self, sample: EvalSample) -> MetricResult:
        """Evaluates a single RAG query-answer sample."""
        f_score = self.compute_faithfulness(sample.contexts, sample.generated_answer)
        r_score = self.compute_answer_relevance(sample.query, sample.generated_answer)
        rec_score = self.compute_context_recall(sample.contexts, sample.ground_truth)
        prec_score = self.compute_context_precision(sample.contexts, sample.ground_truth)

        passed = (f_score >= self.pass_threshold) and (r_score >= self.pass_threshold)
        return MetricResult(
            faithfulness=f_score,
            answer_relevance=r_score,
            context_recall=rec_score,
            context_precision=prec_score,
            passed=passed
        )

    def evaluate_dataset(self, samples: List[EvalSample]) -> EvalReport:
        """Executes a full batch evaluation across a test dataset."""
        start_time = time.time()
        results = []
        
        for sample in samples:
            metrics = self.evaluate_sample(sample)
            results.append({
                "sample_id": sample.sample_id,
                "query": sample.query,
                "metrics": metrics.dict()
            })

        mean_f = round(sum(r["metrics"]["faithfulness"] for r in results) / max(1, len(results)), 4)
        mean_r = round(sum(r["metrics"]["answer_relevance"] for r in results) / max(1, len(results)), 4)
        mean_rec = round(sum(r["metrics"]["context_recall"] for r in results) / max(1, len(results)), 4)
        mean_prec = round(sum(r["metrics"]["context_precision"] for r in results) / max(1, len(results)), 4)
        
        overall = round((mean_f * 0.4) + (mean_r * 0.3) + (mean_rec * 0.15) + (mean_prec * 0.15), 4)
        status = "PASSED" if overall >= self.pass_threshold else "FAILED"
        latency = round((time.time() - start_time) * 1000, 2)

        return EvalReport(
            total_samples=len(samples),
            mean_faithfulness=mean_f,
            mean_answer_relevance=mean_r,
            mean_context_recall=mean_rec,
            mean_context_precision=mean_prec,
            overall_rag_score=overall,
            samples=results,
            execution_time_ms=latency,
            status=status
        )

evaluator = RagasEvaluator(pass_threshold=0.85)
