import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from src.routing.planner import query_planner, TaskType, AgenticQueryPlanner

class TestAgenticQueryPlanner(unittest.TestCase):
    def setUp(self):
        self.planner = AgenticQueryPlanner()

    def test_comparative_query_decomposition(self):
        query = "Compare the operating margins of Q2 versus Q3 and identify cloud spend drivers"
        plan = self.planner.decompose(query)

        self.assertTrue(plan.is_complex)
        self.assertEqual(plan.decomposition_strategy, "PARALLEL_COMPARATIVE_DAG")
        self.assertEqual(len(plan.subgoals), 4)

        # Check subgoals types
        self.assertEqual(plan.subgoals[0].task_type, TaskType.RETRIEVAL)
        self.assertEqual(plan.subgoals[1].task_type, TaskType.RETRIEVAL)
        self.assertEqual(plan.subgoals[2].task_type, TaskType.CALCULATION)
        self.assertEqual(plan.subgoals[3].task_type, TaskType.SYNTHESIS)

        # Check dependencies
        self.assertEqual(plan.subgoals[2].dependencies, ["subgoal_1", "subgoal_2"])
        self.assertEqual(plan.subgoals[3].dependencies, ["subgoal_3"])

        # Check parallel execution layers
        self.assertGreaterEqual(len(plan.execution_layers), 3)
        # Layer 0 must contain subgoal_1 and subgoal_2 for parallel execution
        self.assertIn("subgoal_1", plan.execution_layers[0])
        self.assertIn("subgoal_2", plan.execution_layers[0])

        # Latency savings
        self.assertGreater(plan.concurrency_gain_pct, 15.0)
        self.assertLess(plan.parallel_estimated_ms, plan.sequential_estimated_ms)

    def test_composite_analytical_query_decomposition(self):
        query = "Qual o total de custos de infraestrutura e por que o spend aumentou em Q3?"
        plan = self.planner.decompose(query)

        self.assertTrue(plan.is_complex)
        self.assertEqual(plan.decomposition_strategy, "HYBRID_METRIC_AND_REASONING_DAG")
        self.assertEqual(len(plan.subgoals), 3)

        # Verify SQL + Vector Retrieval parallel layer
        types = [sg.task_type for sg in plan.subgoals]
        self.assertIn(TaskType.ANALYTICAL_SQL, types)
        self.assertIn(TaskType.RETRIEVAL, types)
        self.assertIn(TaskType.SYNTHESIS, types)

    def test_atomic_query_decomposition(self):
        query = "O que é MFA?"
        plan = self.planner.decompose(query)

        self.assertFalse(plan.is_complex)
        self.assertEqual(plan.decomposition_strategy, "ATOMIC_RETRIEVAL_DAG")
        self.assertEqual(len(plan.subgoals), 2)

    def test_plan_execution(self):
        query = "Compare Q1 vs Q2 AWS IAM security compliance"
        plan = self.planner.decompose(query)
        result = self.planner.execute_plan(plan)

        self.assertEqual(result.plan_id, plan.plan_id)
        self.assertEqual(result.completed_subgoals, len(plan.subgoals))
        self.assertEqual(result.parallel_batches_executed, len(plan.execution_layers))
        self.assertIn("Consolidated Agentic Synthesis", result.synthesized_answer)
        self.assertEqual(len(result.reconciled_facts), len(plan.subgoals))
        self.assertGreaterEqual(result.confidence_score, 0.90)

if __name__ == "__main__":
    unittest.main()
