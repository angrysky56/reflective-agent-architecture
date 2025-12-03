import unittest
from unittest.mock import MagicMock, patch

import torch

from src.director.director_core import DirectorConfig, DirectorMVP


class TestDirectorObserverIntegration(unittest.TestCase):
    def setUp(self):
        # Mock dependencies
        self.mock_manifold = MagicMock()
        self.mock_manifold.beta = 1.0 # Fix for f-string formatting
        self.mock_embedding_fn = MagicMock()

        # Patch RecursiveObserver inside director_core
        self.patcher = patch('src.director.director_core.RecursiveObserver')
        self.MockObserverClass = self.patcher.start()
        self.mock_observer_instance = self.MockObserverClass.return_value

        # Initialize Director
        self.config = DirectorConfig(enable_reflexive_closure=False)
        self.director = DirectorMVP(
            manifold=self.mock_manifold,
            config=self.config,
            embedding_fn=self.mock_embedding_fn
        )

    def tearDown(self):
        self.patcher.stop()

    def test_initialization(self):
        """Verify observer is initialized with Director."""
        self.MockObserverClass.assert_called_once()
        self.assertEqual(self.director.observer, self.mock_observer_instance)

    def test_check_and_search_observes_entropy(self):
        """Verify check_and_search logs entropy to observer."""
        # Setup mock entropy check
        self.director.check_entropy = MagicMock(return_value=(False, 0.5))

        # Call check_and_search
        self.director.check_and_search(
            current_state=torch.tensor([1.0]),
            processor_logits=torch.tensor([0.1])
        )

        # Verify observe was called
        self.mock_observer_instance.observe.assert_called()
        # Check arguments of first call
        args, kwargs = self.mock_observer_instance.observe.call_args_list[0]
        self.assertIn("Monitoring entropy", args[0])
        self.assertEqual(kwargs['level'], 0)

    def test_clash_triggers_observation(self):
        """Verify clash detection triggers a level-1 observation."""
        # Setup mock entropy check to return clash
        self.director.check_entropy = MagicMock(return_value=(True, 1.5))
        self.director.monitor.get_threshold = MagicMock(return_value=1.0)

        # Call check_and_search
        self.director.check_and_search(
            current_state=torch.tensor([1.0]),
            processor_logits=torch.tensor([0.1])
        )

        # Verify observe was called for clash
        # We expect multiple calls, find the one with level=1
        clash_observation_found = False
        for call in self.mock_observer_instance.observe.call_args_list:
            args, kwargs = call
            if kwargs.get('level') == 1 and "Clash detected" in args[0]:
                clash_observation_found = True
                break

        self.assertTrue(clash_observation_found, "Clash observation not found")

    def test_high_entropy_triggers_reflection(self):
        """Verify critical entropy triggers reflection."""
        # Setup mock entropy check to return critical entropy
        self.director.check_entropy = MagicMock(return_value=(True, 3.0))

        # Call check_and_search
        self.director.check_and_search(
            current_state=torch.tensor([1.0]),
            processor_logits=torch.tensor([0.1])
        )

        # Verify reflect was called
        self.mock_observer_instance.reflect.assert_called_once()

if __name__ == '__main__':
    unittest.main()
