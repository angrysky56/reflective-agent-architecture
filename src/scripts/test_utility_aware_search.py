import unittest
from unittest.mock import MagicMock

import torch

from src.director.utility_aware_search import UtilityAwareSearch


class TestUtilityAwareSearch(unittest.TestCase):
    def setUp(self) -> None:
        self.search = UtilityAwareSearch(lambda_val=0.5, temperature=1.0)
        # Mock active goals
        self.search.active_goals = {"test_goal": {"utility": 0.8, "description": "A test goal"}}

    def test_valence_modulation(self) -> None:
        """Test simple valence modulation."""
        # Positive valence -> lowered energy (attraction)
        e_pos = self.search.compute_biased_energy(base_energy=10.0, valence=0.5)
        self.assertTrue(e_pos < 10.0)

        # Negative valence -> increased energy (repulsion)
        e_neg = self.search.compute_biased_energy(base_energy=10.0, valence=-0.5)
        self.assertTrue(e_neg > 10.0)

        # Neutral -> no change (if goal_alignment is 0)
        e_neu = self.search.compute_biased_energy(base_energy=10.0, valence=0.0, goal_alignment=0.0)
        self.assertAlmostEqual(e_neu, 10.0)

    def test_clamping(self) -> None:
        """Verify energy doesn't explode."""
        result = self.search.compute_biased_energy(100.0, 1.0)
        self.assertIsInstance(result, float)
        self.assertNotEqual(result, float("inf"))

    def test_lambda_config(self) -> None:
        """Test config impact."""
        s2 = UtilityAwareSearch(lambda_val=0.0)  # No valence effect
        e = s2.compute_biased_energy(10.0, 1.0)
        # Should be close to 10 if lambda is 0 (biased_energy = base - lambda*utility)
        self.assertAlmostEqual(e, 10.0)


if __name__ == "__main__":
    unittest.main()
