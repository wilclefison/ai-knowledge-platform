import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from src.inference.speculative_cascade import (
    SpeculativeCascadeEngine,
    ModelTier,
    SpeculativeTokenResult,
    CascadeInferenceResponse
)


class TestSpeculativeCascade(unittest.TestCase):
    def setUp(self):
        self.engine = SpeculativeCascadeEngine()

    def test_tier1_draft_routing(self):
        query = "What is the port number for PostgreSQL?"
        res = self.engine.generate(query)
        self.assertIsInstance(res, CascadeInferenceResponse)
        self.assertEqual(res.route.selected_tier, ModelTier.TIER_1_DRAFT)
        self.assertGreater(res.route.confidence_score, 0.90)
        self.assertLess(res.total_latency_ms, 50.0)
        self.assertGreater(res.cost_savings_pct, 90.0)

    def test_tier2_mid_routing(self):
        query = "Explain the difference between BM25 sparse search and dense HNSW retrieval"
        res = self.engine.generate(query)
        self.assertEqual(res.route.selected_tier, ModelTier.TIER_2_MID)
        self.assertIn("Llama-3.1-8B", res.route.model_name)
        self.assertGreater(res.route.confidence_score, 0.80)

    def test_tier3_frontier_with_speculative_decoding(self):
        query = "Conduct an in-depth security risk audit of Aurora PostgreSQL database and SOC2 compliance controls"
        res = self.engine.generate(query, use_speculative=True)
        self.assertEqual(res.route.selected_tier, ModelTier.TIER_3_FRONTIER)
        self.assertIsNotNone(res.speculative_stats)
        self.assertIsInstance(res.speculative_stats, SpeculativeTokenResult)
        self.assertGreaterEqual(res.speculative_stats.speculative_speedup_factor, 2.0)
        self.assertGreaterEqual(res.speculative_stats.acceptance_rate_pct, 70.0)
        self.assertIn("Llama-3.3-70B", res.answer)

    def test_benchmark_speculative_decoding_direct(self):
        stats = self.engine.benchmark_speculative_decoding("Generate enterprise audit summary", total_tokens=100)
        self.assertEqual(stats.draft_tokens_proposed, 100)
        self.assertGreater(stats.verified_tokens_accepted, 70)
        self.assertGreater(stats.forward_passes_saved, 50)
        self.assertGreater(stats.speculative_speedup_factor, 2.0)
        self.assertLess(stats.latency_with_speculative_ms, stats.latency_without_speculative_ms)


if __name__ == "__main__":
    unittest.main()
