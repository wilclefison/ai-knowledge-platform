import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from src.retrieval.self_healing_index import (
    SelfHealingIndexEngine,
    IndexHealthStatus,
    IndexMetricReport,
    SelfHealingResult,
    ShardingPolicyResult
)


class TestSelfHealingIndex(unittest.TestCase):
    def setUp(self):
        self.engine = SelfHealingIndexEngine()

    def test_initial_index_health_nominal(self):
        report = self.engine.inspect_index_health("idx_documents_embedding_hnsw")
        self.assertIsInstance(report, IndexMetricReport)
        self.assertEqual(report.status, IndexHealthStatus.HEALTHY)
        self.assertLess(report.fragmentation_percentage, 10.0)
        self.assertLess(report.p99_latency_ms, 20.0)
        self.assertGreater(report.graph_connectivity_score, 0.95)

    def test_simulate_index_degradation(self):
        degraded = self.engine.simulate_degradation(
            index_name="idx_documents_embedding_hnsw",
            dead_tuples_ratio=0.38,
            recall_drift=0.24,
            fragmentation=41.5,
            p99_latency_ms=52.4
        )
        self.assertEqual(degraded.status, IndexHealthStatus.CRITICAL)
        self.assertGreaterEqual(degraded.fragmentation_percentage, 35.0)
        self.assertIn("CRITICAL", degraded.recommendation)

    def test_auto_heal_index_zero_downtime(self):
        # 1. Degrade index
        self.engine.simulate_degradation(
            index_name="idx_documents_embedding_hnsw",
            dead_tuples_ratio=0.40,
            recall_drift=0.25,
            fragmentation=42.0
        )
        # 2. Heal
        result = self.engine.auto_heal_index("idx_documents_embedding_hnsw")
        self.assertIsInstance(result, SelfHealingResult)
        self.assertTrue(result.atomic_swap_success)
        self.assertTrue(result.plan.zero_downtime_guarantee)
        self.assertEqual(result.health_before.status, IndexHealthStatus.CRITICAL)
        self.assertEqual(result.health_after.status, IndexHealthStatus.HEALTHY)
        self.assertLess(result.health_after.fragmentation_percentage, 5.0)
        self.assertLess(result.health_after.recall_drift, 0.02)
        self.assertIn("Successfully executed zero-downtime shadow reindex", result.recovery_summary)

    def test_sharding_policy_isolation(self):
        # High volume tenant -> Dedicated partition
        isolated = self.engine.evaluate_sharding_policy(tenant_id="tenant_fintech_corp", vector_count=350000)
        self.assertIsInstance(isolated, ShardingPolicyResult)
        self.assertEqual(isolated.action_required, "PROVISION_ISOLATED_TENANT_SHARD")
        self.assertTrue(isolated.allocated_shard.is_isolated)
        self.assertEqual(isolated.allocated_shard.m_neighbors, 32)
        self.assertEqual(isolated.allocated_shard.ef_construction, 128)

        # Standard tenant -> Shared pool
        shared = self.engine.evaluate_sharding_policy(tenant_id="tenant_small_corp", vector_count=15000)
        self.assertEqual(shared.action_required, "ASSIGN_SHARED_TENANT_POOL")
        self.assertFalse(shared.allocated_shard.is_isolated)
        self.assertEqual(shared.allocated_shard.m_neighbors, 16)


if __name__ == "__main__":
    unittest.main()
