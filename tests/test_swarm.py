import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from src.agents.swarm import MultiAgentSwarm, SwarmExecutionResult, LeaderElectionResult


class TestMultiAgentSwarm(unittest.TestCase):
    def setUp(self):
        self.swarm = MultiAgentSwarm()

    def test_swarm_agent_registration(self):
        self.assertGreaterEqual(len(self.swarm.agents), 4)
        agent_ids = list(self.swarm.agents.keys())
        self.assertIn("agent_sec_lead", agent_ids)
        self.assertIn("agent_perf_arch", agent_ids)
        self.assertIn("agent_compliance", agent_ids)
        self.assertIn("agent_finops", agent_ids)

    def test_dynamic_leader_election_security(self):
        task = "Evaluate AWS IAM MFA security policies and zero-trust credentials"
        election = self.swarm.elect_leader(task)

        self.assertIsInstance(election, LeaderElectionResult)
        self.assertEqual(election.elected_leader_id, "agent_sec_lead")
        self.assertIn("Principal Security Auditor", election.leader_role)
        self.assertGreater(election.fitness_scores["agent_sec_lead"], election.fitness_scores["agent_finops"])
        self.assertIn("elected Leader", election.election_rationale)

    def test_dynamic_leader_election_compliance(self):
        task = "Review SOC2 Type II regulatory compliance audit standards"
        election = self.swarm.elect_leader(task)

        self.assertEqual(election.elected_leader_id, "agent_compliance")
        self.assertIn("Compliance Officer", election.leader_role)

    def test_swarm_execution_map_reduce(self):
        task = "Audit Aurora PostgreSQL database security, latency, and SOC2 compliance"
        result = self.swarm.execute_swarm(task)

        self.assertIsInstance(result, SwarmExecutionResult)
        self.assertEqual(result.task, task)
        self.assertEqual(result.participating_agents_count, 3)
        self.assertGreater(len(result.worker_results), 0)

        # Verify workers executed
        roles = [w.role for w in result.worker_results]
        self.assertIn("Principal Security Auditor", roles)
        self.assertIn("Performance & Database Architect", roles)
        self.assertIn("Regulatory Compliance Officer", roles)

        # Verify reduce / consensus synthesis
        self.assertIn("FEDERATED SWARM CONSENSUS REPORT", result.consensus_synthesis)
        self.assertGreater(len(result.contradictions_resolved), 0)
        self.assertGreater(result.total_swarm_latency_ms, 0)


if __name__ == "__main__":
    unittest.main()
