import sys
from unittest.mock import MagicMock

# Mock dependencies to avoid ImportError from transformers/huggingface-hub
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["src.compass.self_discover_engine"] = MagicMock()
sys.modules["src.compass.omcd_controller"] = MagicMock()
sys.modules["src.compass.advisors"] = MagicMock()
sys.modules["src.compass.utils"] = MagicMock()

import unittest

from src.compass.config import ExecutiveControllerConfig, SelfDiscoverConfig, oMCDConfig
from src.compass.executive_controller import ExecutiveController


class TestGodelCheck(unittest.TestCase):
    def setUp(self):
        self.config = ExecutiveControllerConfig()
        self.omcd_config = oMCDConfig()
        self.self_discover_config = SelfDiscoverConfig()
        self.advisor_registry = MagicMock()
        self.logger = MagicMock()

        self.controller = ExecutiveController(
            self.config,
            self.omcd_config,
            self.self_discover_config,
            self.advisor_registry,
            self.logger
        )

        # Mock sub-controllers
        self.controller.omcd = MagicMock()
        self.controller.self_discover = MagicMock()
        self.controller.omcd.config = self.omcd_config # Restore config access

    def test_godel_check_triggers(self):
        # Setup conditions for Gödel Loop
        # Director wants to allocate resources (amount > 0.1)
        allocation = {"amount": 0.5}
        # Reality says impossible (score < 0.4)
        solvability = {"solvability_score": 0.2}

        # Cycle 1
        result = self.controller._check_godel_loop(allocation, solvability)
        self.assertFalse(result)
        self.assertEqual(self.controller.godel_loop_count, 1)

        # Cycle 2
        result = self.controller._check_godel_loop(allocation, solvability)
        self.assertFalse(result)
        self.assertEqual(self.controller.godel_loop_count, 2)

        # Cycle 3 (Threshold reached)
        result = self.controller._check_godel_loop(allocation, solvability)
        self.assertTrue(result)
        self.assertEqual(self.controller.godel_loop_count, 3)

    def test_godel_check_resets(self):
        # Setup conditions for Gödel Loop
        allocation = {"amount": 0.5}
        solvability = {"solvability_score": 0.2}

        # Cycle 1
        self.controller._check_godel_loop(allocation, solvability)
        self.assertEqual(self.controller.godel_loop_count, 1)

        # Cycle 2: Reality improves
        solvability_good = {"solvability_score": 0.8}
        result = self.controller._check_godel_loop(allocation, solvability_good)

        self.assertFalse(result)
        self.assertEqual(self.controller.godel_loop_count, 0)

if __name__ == '__main__':
    unittest.main()
