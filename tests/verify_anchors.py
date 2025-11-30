import os
import sys
import unittest
from unittest.mock import MagicMock

import torch

# Add project root to path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)

# Mock dependencies
sys.modules["neo4j"] = MagicMock()
sys.modules["chromadb"] = MagicMock()
sys.modules["chromadb.config"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()
# sys.modules["torch"] = MagicMock() # Use real torch
sys.modules["ollama"] = MagicMock()
sys.modules["src.compass.adapters"] = MagicMock()
sys.modules["src.compass.compass_framework"] = MagicMock()

from src.director.director_core import DirectorConfig, DirectorMVP


class TestAnchorDeployment(unittest.TestCase):
    def test_anchor_on_clash(self):
        # Setup mocks
        mock_manifold = MagicMock()
        mock_manifold.beta = 10.0
        mock_manifold.compute_adaptive_beta.return_value = 5.0

        mock_continuity_service = MagicMock()

        # Initialize Director
        director = DirectorMVP(
            manifold=mock_manifold,
            config=DirectorConfig(),
            continuity_service=mock_continuity_service
        )

        # Mock search result
        mock_search_result = MagicMock()
        mock_search_result.best_pattern = torch.tensor([0.1, 0.2, 0.3])
        mock_search_result.selection_score = 0.9

        # Mock search method to return result
        director.search = MagicMock(return_value=mock_search_result)

        # Mock check_entropy to return clash
        director.check_entropy = MagicMock(return_value=(True, 2.5)) # High entropy

        # Mock compass
        director.compass = MagicMock()
        director.compass.omcd_controller.determine_resource_allocation.return_value = {"amount": 50.0, "confidence": 0.5}

        # Call check_and_search
        current_state = torch.tensor([0.0, 0.0, 0.0])
        logits = torch.tensor([[0.1, 0.9]])

        _ = director.check_and_search(current_state, logits)

        # Verify search was called
        director.search.assert_called()

        # Verify anchor was called
        mock_continuity_service.add_anchor.assert_called_once()

        # Verify arguments
        args, kwargs = mock_continuity_service.add_anchor.call_args
        self.assertTrue("metadata" in kwargs)
        self.assertEqual(kwargs["metadata"]["trigger"], "clash")
        print("Anchor correctly deployed on clash!")

if __name__ == "__main__":
    unittest.main()
