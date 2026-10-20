import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from src.agents.react_agent import ReActAgent, ReActExecutionResult, ReActStep


class TestReActAgent(unittest.TestCase):
    def setUp(self):
        self.agent = ReActAgent(max_iterations=5)

    def test_default_tools_registered(self):
        tools = self.agent.get_available_tools()
        self.assertGreaterEqual(len(tools), 3)
        tool_names = [t.name for t in tools]
        self.assertIn("vector_cluster_inspect", tool_names)
        self.assertIn("knowledge_graph_lookup", tool_names)
        self.assertIn("compliance_rule_verify", tool_names)

    def test_individual_tool_execution(self):
        res_vector = self.agent.tools["vector_cluster_inspect"]({"topic": "iam"})
        self.assertIn("Cluster", res_vector)
        self.assertIn("cosine", res_vector)

        res_graph = self.agent.tools["knowledge_graph_lookup"]({"entity": "aws_iam"})
        self.assertIn("Graph Triples", res_graph)
        self.assertIn("ENFORCES", res_graph)

        res_comp = self.agent.tools["compliance_rule_verify"]({"component": "database"})
        self.assertIn("Compliance Verification", res_comp)
        self.assertIn("SOC2", res_comp)

    def test_react_autonomous_loop(self):
        query = "How does IAM MFA protect the production database under SOC2?"
        result = self.agent.run(query)

        self.assertIsInstance(result, ReActExecutionResult)
        self.assertEqual(result.query, query)
        self.assertTrue(result.is_success)
        self.assertFalse(result.loop_detected)
        self.assertGreaterEqual(result.total_iterations, 4)
        self.assertGreater(len(result.tools_called), 2)
        self.assertIn("vector_cluster_inspect", result.tools_called)
        self.assertIn("knowledge_graph_lookup", result.tools_called)
        self.assertIn("compliance_rule_verify", result.tools_called)

        # Check step structure
        for step in result.steps:
            self.assertIsInstance(step, ReActStep)
            self.assertGreater(len(step.thought), 5)
            self.assertIsNotNone(step.observation)

        self.assertIn("Autonomous ReAct Analysis", result.final_answer)
        self.assertGreater(result.total_latency_ms, 0)

    def test_custom_tool_registration(self):
        agent = ReActAgent()
        agent.register_tool(
            name="cache_latency_checker",
            description="Checks Redis vector cache p99 latency",
            func=lambda args: "Redis VSS p99 latency is 1.8ms",
            schema={"type": "object", "properties": {"node": {"type": "string"}}}
        )
        self.assertIn("cache_latency_checker", agent.tools)
        output = agent.tools["cache_latency_checker"]({"node": "primary"})
        self.assertIn("1.8ms", output)


if __name__ == "__main__":
    unittest.main()
