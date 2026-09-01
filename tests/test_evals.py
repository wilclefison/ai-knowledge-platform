import pytest
from src.evals.ragas_evaluator import RagasEvaluator, EvalSample

def test_evaluator_perfect_faithfulness():
    evaluator = RagasEvaluator()
    sample = EvalSample(
        sample_id="test_01",
        query="What is the MFA policy?",
        contexts=["All privileged accounts must use multi-factor authentication for console and API access."],
        generated_answer="All privileged accounts must use multi-factor authentication for console access.",
        ground_truth="Privileged accounts require MFA."
    )
    result = evaluator.evaluate_sample(sample)
    
    assert result.faithfulness >= 0.85
    assert result.answer_relevance >= 0.80
    assert result.passed is True

def test_evaluator_detects_hallucination():
    evaluator = RagasEvaluator()
    # Context does NOT mention 5 years or $1000 penalty
    sample = EvalSample(
        sample_id="test_02",
        query="What is the penalty for late filing?",
        contexts=["Late filing results in a formal written warning and internal audit review."],
        generated_answer="Late filing results in immediate termination and a five year prison sentence with ten thousand dollars fine.",
        ground_truth="Late filing leads to a written warning."
    )
    result = evaluator.evaluate_sample(sample)
    
    # Faithfulness score drops because claims are not supported by the context
    assert result.faithfulness < 0.60
    assert result.passed is False
