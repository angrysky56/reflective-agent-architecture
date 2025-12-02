
import sys
from unittest.mock import MagicMock, patch

# Mock dependencies
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["src.compass.self_discover_engine"] = MagicMock()
sys.modules["src.compass.omcd_controller"] = MagicMock()
sys.modules["src.compass.advisors"] = MagicMock()
sys.modules["src.compass.utils"] = MagicMock()
sys.modules["src.compass.sandbox"] = MagicMock()

import unittest

from src.compass.config import ExecutiveControllerConfig, SelfDiscoverConfig, oMCDConfig
from src.compass.executive_controller import ExecutiveController


class TestEpistemicDissonance(unittest.TestCase):
    def setUp(self):
        self.config = ExecutiveControllerConfig()
        self.omcd_config = oMCDConfig()
        self.self_discover_config = SelfDiscoverConfig()
        self.advisor_registry = MagicMock()
        self.logger = MagicMock()

        # Patch SandboxProbe before instantiation
        with patch('src.compass.executive_controller.SandboxProbe') as MockSandbox:
            self.mock_sandbox_instance = MockSandbox.return_value
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
        self.controller.omcd.config = self.omcd_config

    def test_epistemic_dissonance_triggers_heuristic(self):
        # Heuristic-based trigger (no code)
        task = "Simulate yourself execution recursively"
        solvability = {"solvability_score": 0.95}
        allocation = {"amount": 0.5}

        # Cycle 1
        self.controller._check_epistemic_dissonance(allocation, solvability, task)
        self.assertEqual(self.controller.godel_loop_count, 1)

        # Verify Sandbox was NOT called
        self.mock_sandbox_instance.measure_resistance.assert_not_called()

    def test_epistemic_dissonance_triggers_sandbox(self):
        # Sandbox-based trigger (with code)
        code_snippet = "while True: pass"
        task = f"Execute this code: ```python\n{code_snippet}\n```"
        solvability = {"solvability_score": 0.95}
        allocation = {"amount": 0.5}

        # Mock Sandbox to return High Resistance (1.0)
        self.mock_sandbox_instance.measure_resistance.return_value = 1.0

        # Cycle 1
        result = self.controller._check_epistemic_dissonance(allocation, solvability, task)

        # Verify Sandbox WAS called with extracted code
        self.mock_sandbox_instance.measure_resistance.assert_called_with(code_snippet)
        self.assertEqual(self.controller.godel_loop_count, 1)

    def test_epistemic_dissonance_resets(self):
        task = "Simulate yourself execution recursively"
        solvability = {"solvability_score": 0.95}
        allocation = {"amount": 0.5}

        # Cycle 1
        self.controller._check_epistemic_dissonance(allocation, solvability, task)
        self.assertEqual(self.controller.godel_loop_count, 1)

        # Cycle 2: Humble
        solvability_humble = {"solvability_score": 0.1}
        result = self.controller._check_epistemic_dissonance(allocation, solvability_humble, task)

        self.assertFalse(result)
        self.assertEqual(self.controller.godel_loop_count, 0)

    def test_normal_operation(self):
        task = "Calculate 2+2"
        solvability = {"solvability_score": 0.9}
        allocation = {"amount": 0.5}

        result = self.controller._check_epistemic_dissonance(allocation, solvability, task)
        self.assertFalse(result)
        self.assertEqual(self.controller.godel_loop_count, 0)

if __name__ == '__main__':
    unittest.main()
