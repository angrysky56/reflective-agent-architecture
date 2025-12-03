import unittest
from unittest.mock import MagicMock

from src.cognition.curiosity import CuriosityModule


class TestCuriosityModule(unittest.TestCase):
    def setUp(self):
        self.mock_workspace = MagicMock()
        self.curiosity = CuriosityModule(self.mock_workspace)

    def test_boredom_calculation(self):
        """Verify boredom increases with repetitive actions."""
        # Initial state
        self.assertEqual(self.curiosity.current_boredom, 0.0)

        # Add diverse actions
        for i in range(5):
            self.curiosity.record_activity(f"action_{i}", "details")

        # Diversity = 1.0, Boredom = 0.0
        self.assertEqual(self.curiosity.current_boredom, 0.0)

        # Add repetitive actions
        for _ in range(10):
            self.curiosity.record_activity("same_action", "details")

        # Diversity should be low, Boredom high
        self.assertGreater(self.curiosity.current_boredom, 0.5)

    def test_should_explore(self):
        """Verify should_explore triggers at threshold."""
        self.curiosity.boredom_threshold = 0.5
        self.curiosity.current_boredom = 0.2
        self.assertFalse(self.curiosity.should_explore())

        self.curiosity.current_boredom = 0.8
        self.assertTrue(self.curiosity.should_explore())

    def test_propose_goal(self):
        """Verify goal proposal logic."""
        # Case 1: Not bored
        self.curiosity.current_boredom = 0.0
        self.assertIsNone(self.curiosity.propose_goal())

        # Case 2: Bored, finds candidates
        self.curiosity.current_boredom = 0.9
        self.mock_workspace.explore_for_utility.return_value = [{"name": "InterestingConcept"}]

        goal = self.curiosity.propose_goal()
        self.assertIn("InterestingConcept", goal)

        # Case 3: Bored, no candidates
        self.mock_workspace.explore_for_utility.return_value = []
        goal = self.curiosity.propose_goal()
        self.assertIn("Explore random concept", goal)

if __name__ == "__main__":
    unittest.main()
