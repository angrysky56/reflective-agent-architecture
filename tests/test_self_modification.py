import unittest
from unittest.mock import MagicMock, patch

import torch

from src.director.director_core import DirectorConfig, DirectorMVP


class TestSelfModification(unittest.TestCase):
    def setUp(self):
        # Mock dependencies
        self.mock_manifold = MagicMock()
        self.mock_manifold.beta = 1.0
        self.mock_embedding_fn = MagicMock()

        # Patch RecursiveObserver inside director_core
        self.patcher = patch('src.director.director_core.RecursiveObserver')
        self.MockObserverClass = self.patcher.start()
        self.mock_observer_instance = self.MockObserverClass.return_value

        # Initialize Director
        self.config = DirectorConfig(enable_reflexive_closure=False, search_metric="cosine")
        self.director = DirectorMVP(
            manifold=self.mock_manifold,
            config=self.config,
            embedding_fn=self.mock_embedding_fn
        )

    def tearDown(self):
        self.patcher.stop()

    def test_switch_strategy_action(self):
        """Verify SWITCH_STRATEGY action updates director config."""
        # Setup mock entropy check to return critical entropy
        self.director.check_entropy = MagicMock(return_value=(True, 3.0))

        # Setup mock reflection response
        action = {
            "observation": "Agent is stuck in a loop.",
            "action_type": "SWITCH_STRATEGY",
            "parameters": {"metric": "euclidean"}
        }
        self.mock_observer_instance.reflect.return_value = action

        # Verify initial state
        self.assertEqual(self.director.config.search_metric, "cosine")

        # Call check_and_search
        self.director.check_and_search(
            current_state=torch.tensor([1.0]),
            processor_logits=torch.tensor([0.1])
        )

        # Verify config update
        self.assertEqual(self.director.config.search_metric, "euclidean")

if __name__ == '__main__':
    unittest.main()
