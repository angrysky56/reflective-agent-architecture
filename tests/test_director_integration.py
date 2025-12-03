
import unittest
from unittest.mock import MagicMock, patch

import torch

from src.director.director_core import DirectorConfig, DirectorMVP
from src.director.hybrid_search import HybridSearchConfig, HybridSearchStrategy
from src.director.reflexive_closure_engine import ReflexiveClosureEngine


class TestDirectorIntegration(unittest.TestCase):
    def setUp(self):
        self.mock_manifold = MagicMock()
        self.mock_manifold.beta = 1.0
        self.mock_manifold.compute_adaptive_beta.return_value = 5.0
        self.mock_manifold.get_patterns.return_value = torch.randn(10, 64)

        self.config = DirectorConfig(
            search_k=5,
            search_metric="cosine",
            enable_reflexive_closure=True
        )

        self.director = DirectorMVP(
            manifold=self.mock_manifold,
            config=self.config,
            embedding_fn=lambda x: torch.randn(64)
        )

        # Mock Reflexive Engine
        self.director.reflexive_engine = MagicMock(spec=ReflexiveClosureEngine)

    def test_get_remedial_action(self):
        """Test adaptive advice generation."""
        # Case 1: Looping
        action = self.director.get_remedial_action("Looping", -0.9, 0.5)
        self.assertIn("Stop", action["advice"])
        self.assertIn("WARNING", action["warnings"][0])

        # Case 2: Confused
        action = self.director.get_remedial_action("Confused", -0.5, 0.9)
        self.assertIn("deconstruct", action["advice"])

        # Case 3: High Entropy
        action = self.director.get_remedial_action("Flow", -0.9, 0.95)
        self.assertIn("High Entropy", action["warnings"][0])

    def test_dynamic_search_parameters(self):
        """Test that check_and_search uses dynamic parameters from Reflexive Engine."""
        # Setup mocks
        self.director.reflexive_engine.get_parameter.side_effect = lambda name, state: {
            "search_k": 10,
            "search_metric": "euclidean"
        }.get(name)

        self.director.search = MagicMock()
        self.director.latest_cognitive_state = ("Flow", -1.0)

        # Trigger check_and_search (force clash)
        # We need to mock monitor.check_logits to return True (clash)
        self.director.monitor.check_logits = MagicMock(return_value=(True, 2.5))

        # Mock search result
        mock_result = MagicMock()
        mock_result.selection_score = 0.95
        mock_result.best_pattern = torch.randn(64)
        self.director.search.return_value = mock_result

        self.director.monitor.get_threshold = MagicMock(return_value=2.0)
        self.director.reflexive_engine.get_threshold = MagicMock(return_value=1.5)

        context = {}
        self.director.check_and_search(torch.randn(64), context)

        # Verify search was called with dynamic params
        self.director.search.assert_called_with(
            unittest.mock.ANY,
            unittest.mock.ANY,
            k=10,
            metric="euclidean"
        )

    def test_hybrid_search_overrides(self):
        """Test that HybridSearchStrategy accepts overrides."""
        hybrid = HybridSearchStrategy(
            manifold=self.mock_manifold,
            ltn_refiner=MagicMock(),
            config=HybridSearchConfig(knn_k=5, knn_metric="cosine")
        )

        # Mock _try_knn_search to verify args
        hybrid._try_knn_search = MagicMock()

        # Call search with overrides
        hybrid.search(
            current_state=torch.randn(64),
            k=20,
            metric="dot"
        )

        # Verify _try_knn_search received overrides
        hybrid._try_knn_search.assert_called_with(
            unittest.mock.ANY,
            unittest.mock.ANY,
            k=20,
            metric="dot"
        )

if __name__ == "__main__":
    unittest.main()
