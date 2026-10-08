import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from src.retrieval.crag_evaluator import crag_evaluator, DocumentGrade, CRAGAction, CorrectiveRAGEvaluator

class TestCorrectiveRAGEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = CorrectiveRAGEvaluator()

    def test_grade_relevant_chunks(self):
        query = "AWS IAM MFA policy requirements"
        chunks = [
            {
                "chunk_id": "c1",
                "text": "AWS IAM policies require multi-factor authentication (MFA) on all administrative credentials."
            },
            {
                "chunk_id": "c2",
                "text": "The cafeteria menu on corporate headquarters serves breakfast from 8 AM to 10 AM daily."
            }
        ]

        report = self.evaluator.grade_documents(query, chunks)

        self.assertEqual(report.total_evaluated, 2)
        self.assertEqual(report.kept_chunks_count, 1)
        self.assertEqual(report.discarded_chunks_count, 1)

        self.assertEqual(report.graded_chunks[0].grade, DocumentGrade.RELEVANT)
        self.assertEqual(report.graded_chunks[1].grade, DocumentGrade.IRRELEVANT)
        self.assertGreater(report.token_reduction_pct, 0.0)

    def test_grade_ambiguous_trigger_rewrite(self):
        query = "Kubernetes ingress controller configuration"
        chunks = [
            {
                "chunk_id": "c1",
                "text": "Configuration settings for web server deployment."
            },
            {
                "chunk_id": "c2",
                "text": "General cloud networking concepts and subnets."
            }
        ]

        report = self.evaluator.grade_documents(query, chunks)
        self.assertIn(report.overall_action, [CRAGAction.AMBIGUOUS_REWRITE, CRAGAction.INCORRECT_FALLBACK])

    def test_all_irrelevant_triggers_incorrect_fallback(self):
        query = "PostgreSQL pgvector index tuning"
        chunks = [
            {
                "chunk_id": "c1",
                "text": "How to make espresso using a manual portafilter."
            }
        ]

        report = self.evaluator.grade_documents(query, chunks)
        self.assertEqual(report.overall_action, CRAGAction.INCORRECT_FALLBACK)
        self.assertTrue(report.fallback_query.startswith("web_search:"))

if __name__ == "__main__":
    unittest.main()
