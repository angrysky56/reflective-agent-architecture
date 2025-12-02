import os
import sys
import unittest
from typing import Any, Dict, List

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv

from src.server import CognitiveWorkspace, CWDConfig

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

class TestDynamicGraphQueries(unittest.TestCase):
    def setUp(self):
        # Configure
        self.config = CWDConfig()
        # Override with env vars if needed, but CWDConfig loads from env

        self.workspace = CognitiveWorkspace(self.config)
        self.driver = self.workspace.neo4j_driver

        # Create test data
        with self.driver.session() as session:
            session.run("MATCH (n:TestNode) DETACH DELETE n")
            session.run("""
                CREATE (a:TestNode {id: 'test_a', name: 'Alpha', value: 10})
                CREATE (b:TestNode {id: 'test_b', name: 'Beta', value: 20})
                CREATE (c:TestNode {id: 'test_c', name: 'Gamma', value: 10})
                CREATE (a)-[:TEST_REL {weight: 0.5}]->(b)
                CREATE (b)-[:TEST_REL {weight: 0.8}]->(c)
                CREATE (a)-[:OTHER_REL]->(c)
            """)

    def tearDown(self):
        # Cleanup
        with self.driver.session() as session:
            session.run("MATCH (n:TestNode) DETACH DELETE n")
        self.workspace.close()

    def test_search_nodes(self):
        print("\nTesting search_nodes...")
        # Test 1: Simple label search
        results = self.workspace.search_nodes("TestNode")
        self.assertEqual(len(results), 3)
        print(f"Found {len(results)} nodes with label 'TestNode'")

        # Test 2: Property filter
        results = self.workspace.search_nodes("TestNode", {"value": 10})
        self.assertEqual(len(results), 2)
        names = [r["n"]["name"] for r in results]
        self.assertIn("Alpha", names)
        self.assertIn("Gamma", names)
        print(f"Found {len(results)} nodes with value=10: {names}")

        # Test 3: Non-existent label
        results = self.workspace.search_nodes("NonExistentLabel")
        self.assertEqual(len(results), 0)
        print("Correctly found 0 nodes for non-existent label")

    def test_traverse_relationships(self):
        print("\nTesting traverse_relationships...")
        # Test 1: Outgoing traversal
        results = self.workspace.traverse_relationships("test_a", "TEST_REL", "OUTGOING")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["m"]["id"], "test_b")
        print("Correctly traversed OUTGOING TEST_REL from A to B")

        # Test 2: Incoming traversal
        results = self.workspace.traverse_relationships("test_b", "TEST_REL", "INCOMING")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["m"]["id"], "test_a")
        print("Correctly traversed INCOMING TEST_REL from B to A")

        # Test 3: Both directions
        results = self.workspace.traverse_relationships("test_b", "TEST_REL", "BOTH")
        # Should find A (incoming) and C (outgoing)
        self.assertEqual(len(results), 2)
        ids = [r["m"]["id"] for r in results]
        self.assertIn("test_a", ids)
        self.assertIn("test_c", ids)
        print(f"Correctly traversed BOTH directions from B: {ids}")

if __name__ == '__main__':
    unittest.main()
