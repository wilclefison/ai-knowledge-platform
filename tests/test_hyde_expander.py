import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from src.retrieval.hyde_expander import HyDEExpander, HyDEResult


class TestHyDEExpander(unittest.TestCase):
    def setUp(self):
        self.expander = HyDEExpander()

    def test_generate_hypothetical_doc_iam(self):
        query = "How to enforce IAM MFA policy on production roles?"
        doc = self.expander.generate_hypothetical_doc(query)
        self.assertIn("AWS Identity and Access Management", doc)
        self.assertIn("MFA", doc)
        self.assertGreater(len(doc), 100)

    def test_generate_hypothetical_doc_database(self):
        query = "What is the Aurora database encryption standard?"
        doc = self.expander.generate_hypothetical_doc(query)
        self.assertIn("Aurora PostgreSQL", doc)
        self.assertIn("AES-256", doc)

    def test_expand_multi_query(self):
        query = "Transit Gateway network isolation"
        perspectives = self.expander.expand_multi_query(query)
        self.assertEqual(len(perspectives), 3)
        names = [p.perspective_name for p in perspectives]
        self.assertIn("Technical Architecture", names)
        self.assertIn("Security & Compliance", names)
        self.assertIn("Operational Runbook", names)
        for p in perspectives:
            self.assertTrue(p.expanded_query.startswith(query))
            self.assertGreater(len(p.focus_rationale), 10)

    def test_process_query_end_to_end(self):
        query = "AWS IAM MFA policy 2024"
        res = self.expander.process_query(query)
        self.assertIsInstance(res, HyDEResult)
        self.assertEqual(res.original_query, query)
        self.assertGreater(len(res.hypothetical_document), 50)
        self.assertEqual(len(res.expanded_queries), 3)
        self.assertGreaterEqual(res.dense_alignment_score, 0.8)
        self.assertGreater(res.vocabulary_overlap_gain_pct, 100.0)
        self.assertGreater(res.latency_ms, 0)
        self.assertIn("HyDE synthesized", res.retrieval_impact_summary)


if __name__ == "__main__":
    unittest.main()
