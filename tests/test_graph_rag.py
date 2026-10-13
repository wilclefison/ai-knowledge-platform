import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from src.retrieval.graph_rag import GraphRAGEngine, EntityNode, RelationshipEdge


class TestGraphRAGEngine(unittest.TestCase):
    def setUp(self):
        self.engine = GraphRAGEngine()

    def test_graph_initial_seed(self):
        self.assertIn("aws_iam", self.engine.nodes)
        self.assertIn("mfa_policy", self.engine.nodes)
        self.assertIn("prod_database", self.engine.nodes)
        self.assertGreaterEqual(len(self.engine.nodes), 6)
        self.assertGreaterEqual(len(self.engine.adjacency["aws_iam"]), 2)

    def test_entity_and_relation_extraction(self):
        sample_text = (
            "Under the 2024 security baseline, AWS IAM enforces mandatory MFA policies "
            "to protect access to the Production Database, which is audited under SOC2 Compliance."
        )
        result = self.engine.extract_entities_and_relations(sample_text, chunk_id="chunk_sec_01")

        self.assertGreaterEqual(len(result.extracted_entities), 3)
        extracted_ids = [e.id for e in result.extracted_entities]
        self.assertIn("aws_iam", extracted_ids)
        self.assertIn("mfa_policy", extracted_ids)
        self.assertIn("prod_database", extracted_ids)

        self.assertGreaterEqual(len(result.extracted_relations), 1)
        self.assertGreaterEqual(result.triples_count, 1)
        self.assertGreaterEqual(result.processing_time_ms, 0)

    def test_multi_hop_traversal(self):
        # 2-hop traversal starting from okta_sso:
        # Okta SSO -> AWS IAM -> MFA Policy -> Production Database
        subgraph = self.engine.multi_hop_traversal(["okta_sso"], max_hops=2)

        node_ids = [n.id for n in subgraph.nodes]
        self.assertIn("okta_sso", node_ids)
        self.assertIn("aws_iam", node_ids)
        self.assertIn("mfa_policy", node_ids)
        self.assertGreaterEqual(len(subgraph.triples), 2)

    def test_graph_query_reasoning(self):
        query = "How does AWS IAM and MFA authentication protect our production database and meet SOC2?"
        res = self.engine.query_graph(query, max_hops=2)

        self.assertIn("aws_iam", res.seed_entities)
        self.assertGreaterEqual(res.traversed_nodes_count, 3)
        self.assertGreaterEqual(len(res.reasoning_paths), 1)
        self.assertIn("STRUCTURED KNOWLEDGE GRAPH CONTEXT", res.synthesized_context)
        self.assertIn("transitive_reasoning", res.graph_vs_vector_analysis)
        self.assertGreater(res.latency_ms, 0)

    def test_graph_schema_summary(self):
        summary = self.engine.get_schema_summary()

        self.assertGreaterEqual(summary.total_nodes, 6)
        self.assertGreaterEqual(summary.total_edges, 5)
        self.assertIn("SYSTEM", summary.entity_types)
        self.assertIn("POLICY", summary.entity_types)
        self.assertGreater(len(summary.top_connected_entities), 0)


if __name__ == "__main__":
    unittest.main()
