import json
import unittest
from unittest.mock import MagicMock, patch

from src.director.recursive_observer import RecursiveObserver, ThoughtNode


class TestRecursiveObserver(unittest.TestCase):
    def setUp(self):
        # Patch LLMFactory to avoid actual API calls
        self.patcher = patch('src.director.recursive_observer.LLMFactory')
        self.mock_factory = self.patcher.start()
        self.mock_llm = MagicMock()
        self.mock_factory.create_provider.return_value = self.mock_llm

        self.observer = RecursiveObserver()

    def tearDown(self):
        self.patcher.stop()

    def test_thought_node_creation(self):
        node = ThoughtNode(content="Test thought", level=0)
        self.assertEqual(node.content, "Test thought")
        self.assertEqual(node.level, 0)
        self.assertIsNotNone(node.id)
        self.assertEqual(node.children, [])

    def test_observe_base_thought(self):
        node = self.observer.observe("Base thought", level=0)
        self.assertEqual(len(self.observer.root_thoughts), 1)
        self.assertEqual(self.observer.root_thoughts[0], node)
        self.assertEqual(len(self.observer.active_context), 1)

    def test_hierarchy_construction(self):
        # Create a base thought
        base = self.observer.observe("Base thought", level=0)

        # Create a meta thought
        meta = self.observer.observe("Meta thought", level=1)

        # Check if meta thought is linked to base thought
        self.assertEqual(len(base.children), 1)
        self.assertEqual(base.children[0], meta)
        self.assertEqual(meta.parent_id, base.id)

    def test_reflect(self):
        # Setup context
        self.observer.observe("I am thinking about X", level=0)
        self.observer.observe("But X implies Y", level=0)

        # Mock LLM response
        self.mock_llm.generate.return_value = "This is a circular argument."

        # Trigger reflection
        response = self.observer.reflect()

        # Verify
        self.assertEqual(response, "This is a circular argument.")
        self.mock_llm.generate.assert_called_once()

        # Check if reflection was added as a thought
        last_thought = self.observer.active_context[-1]
        self.assertIn("REFLECTION: This is a circular argument.", last_thought.content)
        self.assertEqual(last_thought.level, 1) # Should be level 0 + 1

    def test_serialization(self):
        self.observer.observe("Root", level=0)
        json_str = self.observer.get_hierarchy_json()
        data = json.loads(json_str)

        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['content'], "Root")

if __name__ == '__main__':
    unittest.main()
