
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import torch

from src.director.adaptive_criterion import AdaptiveCriterion, ParameterConfig
from src.director.meta_pattern_analyzer import PatternInsight
from src.director.reflexive_closure_engine import ReflexiveClosureEngine


class TestMultiParameterClosure(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.persistence_path = Path(self.test_dir) / "test_criterion.json"

        self.criterion = AdaptiveCriterion(persistence_path=self.persistence_path)
        self.engine = ReflexiveClosureEngine(criterion=self.criterion)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_parameter_initialization(self):
        """Verify all parameters are initialized correctly."""
        params = self.criterion.state.parameters
        self.assertIn("entropy_threshold", params)
        self.assertIn("search_k", params)
        self.assertIn("search_depth", params)
        self.assertIn("search_metric", params)

        self.assertEqual(params["search_k"].value, 5)
        self.assertEqual(params["search_metric"].value, "cosine")

    def test_parameter_updates(self):
        """Verify parameters can be updated via insights."""
        # Update search_k (int)
        insight_k = PatternInsight(
            pattern_type="parameter_optimization",
            confidence=0.9,
            suggested_adjustment={"parameter": "search_k", "value": 10},
            evidence_count=5,
            recommendation="Increase search_k to 10"
        )
        self.criterion.update(insight_k)
        self.assertEqual(self.criterion.get_parameter("search_k"), 10)

        # Update entropy_threshold (float)
        insight_thresh = PatternInsight(
            pattern_type="parameter_optimization",
            confidence=0.9,
            suggested_adjustment={"parameter": "entropy_threshold", "multiplier": 1.1},
            evidence_count=5,
            recommendation="Increase entropy threshold"
        )
        # Default is 2.0, 2.0 * 1.1 = 2.2
        self.criterion.update(insight_thresh)
        self.assertAlmostEqual(self.criterion.get_parameter("entropy_threshold"), 2.2)

    def test_type_aware_exploration(self):
        """Verify engine explores differently based on type."""
        # Force exploration
        self.engine.exploration_rate = 1.0

        # Float exploration (entropy_threshold)
        val_float = self.engine.get_parameter("entropy_threshold")
        self.assertIsInstance(val_float, float)
        # Should be perturbed from 2.0 (or 2.2 if state persisted, but new instance here)
        # We can't guarantee exact value due to randomness, but it should be float

        # Int exploration (search_k)
        val_int = self.engine.get_parameter("search_k")
        self.assertIsInstance(val_int, int)
        # Should be perturbed from 5

        # Categorical (metric) - currently no exploration implemented, should return base
        val_str = self.engine.get_parameter("search_metric")
        self.assertEqual(val_str, "cosine")

    def test_state_specific_overrides(self):
        """Verify parameters can have state-specific overrides."""
        # Set override for "Looping" state
        self.criterion.state.parameters["search_k"].state_overrides["Looping"] = 20

        # Normal state
        self.assertEqual(self.criterion.get_parameter("search_k", "Flow"), 5)

        # Looping state
        self.assertEqual(self.criterion.get_parameter("search_k", "Looping"), 20)

    def test_co_adaptation_simulation(self):
        """Simulate a scenario where high entropy leads to parameter adaptation."""
        # 1. Initial state
        self.assertEqual(self.criterion.get_parameter("search_k"), 5)

        # 2. Simulate intervention failure with low k
        # Analyzer would generate insight to increase k
        insight = PatternInsight(
            pattern_type="parameter_optimization",
            confidence=0.8,
            suggested_adjustment={"parameter": "search_k", "value": 8},
            evidence_count=3,
            recommendation="Increase search_k for better coverage"
        )

        # 3. Apply update
        self.engine.criterion.update(insight)

        # 4. Verify adaptation
        self.assertEqual(self.engine.get_parameter("search_k"), 8)

        # 5. Verify persistence
        self.criterion._save()

        # Reload
        new_criterion = AdaptiveCriterion(persistence_path=self.persistence_path)
        self.assertEqual(new_criterion.get_parameter("search_k"), 8)

if __name__ == "__main__":
    unittest.main()
