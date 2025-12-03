import os
import unittest
from unittest.mock import MagicMock, patch

from src.integration.sleep_cycle import SleepCycle
from src.server import CognitiveWorkspace


class TestRuminator(unittest.TestCase):
    def setUp(self):
        self.mock_workspace = MagicMock(spec=CognitiveWorkspace)
        self.mock_workspace.system_guide = MagicMock()
        self.mock_workspace.neo4j_driver = MagicMock()
        self.mock_session = MagicMock()
        self.mock_workspace.neo4j_driver.session.return_value.__enter__.return_value = self.mock_session

        self.sleep_cycle = SleepCycle(workspace=self.mock_workspace)

    @patch("src.integration.sleep_cycle.time.sleep")
    def test_ruminate_on_codebase(self, mock_sleep):
        # Mock scan result
        self.mock_workspace.system_guide.scan_codebase.return_value = "Scanned."

        # Mock finding undocumented bookmarks
        self.mock_session.run.side_effect = [
            # First call: find undocumented
            [
                {"id": "b:1", "snippet": "def foo(): pass", "file": "f.py", "line": 1},
                {"id": "b:2", "snippet": "class Bar: pass", "file": "f.py", "line": 5}
            ],
            # Subsequent calls: update bookmarks (return nothing)
            [], []
        ]

        # Mock LLM generation via ruminator_provider
        self.mock_workspace.ruminator_provider = MagicMock()
        self.mock_workspace.ruminator_provider.generate.return_value = "Generated Docstring"
        self.mock_workspace.ruminator_provider.model_name = "test-ruminator"

        # Run rumination
        result = self.sleep_cycle._ruminate_on_codebase()

        # Verify
        self.assertEqual(result["status"], "active")
        self.assertEqual(result["processed"], 2)
        self.mock_workspace.system_guide.scan_codebase.assert_called_with(".")
        self.assertEqual(self.mock_workspace.ruminator_provider.generate.call_count, 2)

        # Verify rate limiting sleep called
        self.assertEqual(mock_sleep.call_count, 2)

    def test_ruminate_no_undocumented(self):
        self.mock_workspace.system_guide.scan_codebase.return_value = "Scanned."
        self.mock_session.run.return_value = [] # No results

        result = self.sleep_cycle._ruminate_on_codebase()

        self.assertEqual(result["status"], "idle")
        self.assertEqual(result["message"], "No undocumented code found.")

if __name__ == '__main__':
    unittest.main()
