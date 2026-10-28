import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from src.cache.semantic_cache import semantic_cache
from src.cache.cdc_sync import CDCSyncEngine, CDCOperation, CDCEvent, InvalidationResult


class TestCDCSync(unittest.TestCase):
    def setUp(self):
        self.cdc_engine = CDCSyncEngine()
        # Seed test items in semantic cache
        semantic_cache._cache_store = []
        semantic_cache.store(
            tenant_id="tenant_acme_corp",
            query="What are the IAM requirements for security in 2024?",
            answer="MFA is mandatory on all admin roles.",
            citations=[{
                "document_title": "AWS Security Whitepaper",
                "chunk_id": "chunk_iam_001",
                "tenant_id": "tenant_acme_corp"
            }],
            tokens_saved=850
        )
        semantic_cache.store(
            tenant_id="tenant_acme_corp",
            query="What is the Aurora PostgreSQL backup retention policy?",
            answer="Automated daily backups retained for 35 days.",
            citations=[{
                "document_title": "Database Ops Manual",
                "chunk_id": "chunk_aurora_999",
                "tenant_id": "tenant_acme_corp"
            }],
            tokens_saved=620
        )

    def test_cdc_insert_event_noop(self):
        event = CDCEvent(
            event_id="evt_ins_01",
            op=CDCOperation.INSERT,
            tenant_id="tenant_acme_corp",
            chunk_id="chunk_new_doc",
            lsn=104859000
        )
        result = self.cdc_engine.process_cdc_event(event)
        self.assertEqual(result.op, CDCOperation.INSERT)
        self.assertEqual(result.invalidated_count, 0)
        self.assertEqual(len(semantic_cache._cache_store), 2)
        self.assertEqual(result.cache_hit_rate_preserved_pct, 100.0)

    def test_cdc_update_surgical_invalidation(self):
        # Update chunk_iam_001: should evict ONLY the IAM query, preserving Aurora query
        result = self.cdc_engine.simulate_cdc_mutation(
            tenant_id="tenant_acme_corp",
            chunk_id="chunk_iam_001",
            op=CDCOperation.UPDATE,
            updated_text="AWS IAM now requires WebAuthn FIDO2 keys instead of standard TOTP."
        )

        self.assertIsInstance(result, InvalidationResult)
        self.assertEqual(result.invalidated_count, 1)
        self.assertIn("What are the IAM requirements", result.invalidated_queries[0])

        # Verify the unrelated Aurora cache entry is still alive!
        self.assertEqual(len(semantic_cache._cache_store), 1)
        self.assertEqual(semantic_cache._cache_store[0].citations[0]["chunk_id"], "chunk_aurora_999")
        self.assertGreater(result.cache_hit_rate_preserved_pct, 0.0)

    def test_cdc_delete_invalidation(self):
        result = self.cdc_engine.simulate_cdc_mutation(
            tenant_id="tenant_acme_corp",
            chunk_id="chunk_aurora_999",
            op=CDCOperation.DELETE
        )
        self.assertEqual(result.invalidated_count, 1)
        self.assertIn("Aurora PostgreSQL backup", result.invalidated_queries[0])
        # Only IAM remains
        self.assertEqual(len(semantic_cache._cache_store), 1)
        self.assertEqual(semantic_cache._cache_store[0].citations[0]["chunk_id"], "chunk_iam_001")

    def test_cdc_tenant_isolation(self):
        # Mutation on different tenant should NOT touch tenant_acme_corp cache entries
        result = self.cdc_engine.simulate_cdc_mutation(
            tenant_id="tenant_other_corp",
            chunk_id="chunk_iam_001",
            op=CDCOperation.UPDATE
        )
        self.assertEqual(result.invalidated_count, 0)
        self.assertEqual(len(semantic_cache._cache_store), 2)


if __name__ == "__main__":
    unittest.main()
