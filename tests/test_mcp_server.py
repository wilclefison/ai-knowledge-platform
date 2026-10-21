import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from src.mcp.server import MCPServer, MCP_PROTOCOL_VERSION


class TestMCPServer(unittest.TestCase):
    def setUp(self):
        self.server = MCPServer()

    def test_mcp_initialize(self):
        req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "clientInfo": {"name": "test-ide", "version": "1.0.0"}
            }
        }
        res = self.server.handle_jsonrpc(req)
        self.assertEqual(res["jsonrpc"], "2.0")
        self.assertEqual(res["id"], 1)
        self.assertIn("result", res)
        self.assertEqual(res["result"]["protocolVersion"], MCP_PROTOCOL_VERSION)
        self.assertIn("tools", res["result"]["capabilities"])
        self.assertIn("resources", res["result"]["capabilities"])
        self.assertEqual(res["result"]["serverInfo"]["name"], "ai-knowledge-platform-mcp")

    def test_mcp_tools_list(self):
        req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
        res = self.server.handle_jsonrpc(req)
        tools = res["result"]["tools"]
        self.assertGreaterEqual(len(tools), 3)
        tool_names = [t["name"] for t in tools]
        self.assertIn("search_knowledge_base", tool_names)
        self.assertIn("query_knowledge_graph", tool_names)
        self.assertIn("verify_compliance_controls", tool_names)

    def test_mcp_tools_call(self):
        req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "search_knowledge_base",
                "arguments": {"query": "AWS IAM MFA", "top_k": 3}
            }
        }
        res = self.server.handle_jsonrpc(req)
        self.assertIn("result", res)
        self.assertFalse(res["result"]["isError"])
        self.assertGreater(len(res["result"]["content"]), 0)
        self.assertIn("chunk_sec_01", res["result"]["content"][0]["text"])

    def test_mcp_resources_list_and_read(self):
        req_list = {"jsonrpc": "2.0", "id": 4, "method": "resources/list"}
        res_list = self.server.handle_jsonrpc(req_list)
        resources = res_list["result"]["resources"]
        self.assertGreaterEqual(len(resources), 2)
        uris = [r["uri"] for r in resources]
        self.assertIn("schema://database/aurora-prod", uris)

        req_read = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "resources/read",
            "params": {"uri": "schema://database/aurora-prod"}
        }
        res_read = self.server.handle_jsonrpc(req_read)
        self.assertIn("contents", res_read["result"])
        self.assertEqual(res_read["result"]["contents"][0]["uri"], "schema://database/aurora-prod")

    def test_mcp_prompts_list(self):
        req = {"jsonrpc": "2.0", "id": 6, "method": "prompts/list"}
        res = self.server.handle_jsonrpc(req)
        prompts = res["result"]["prompts"]
        self.assertGreaterEqual(len(prompts), 1)
        self.assertEqual(prompts[0]["name"], "security_audit_review")

    def test_mcp_invalid_method(self):
        req = {"jsonrpc": "2.0", "id": 7, "method": "unsupported/action"}
        res = self.server.handle_jsonrpc(req)
        self.assertIn("error", res)
        self.assertEqual(res["error"]["code"], -32601)


if __name__ == "__main__":
    unittest.main()
