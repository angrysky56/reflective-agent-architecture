import unittest
from unittest.mock import MagicMock

from src.integration.cwd_raa_bridge import BridgeConfig, CWDRAABridge


class TestAutoNap(unittest.TestCase):
    def test_auto_nap_trigger(self):
        # Mock components
        mock_cwd = MagicMock()
        mock_director = MagicMock()
        mock_manifold = MagicMock()
        mock_sleep_cycle = MagicMock()

        # Configure Director to return CRITICAL LOW ENERGY (-0.7)
        mock_director.latest_cognitive_state = ("Exhausted", -0.7)
        mock_director.check_entropy.return_value = (False, 0.1) # No clash, just tired

        # Configure Sleep Cycle
        mock_sleep_cycle.dream.return_value = {"message": "Nap complete"}

        # Initialize Bridge
        bridge = CWDRAABridge(
            cwd_server=mock_cwd,
            raa_director=mock_director,
            manifold=mock_manifold,
            sleep_cycle=mock_sleep_cycle
        )

        # Execute operation
        bridge.execute_monitored_operation("test_op", {})

        # Verify Auto-Nap was triggered
        mock_sleep_cycle.dream.assert_called_once()
        print("Auto-Nap triggered successfully!")

if __name__ == "__main__":
    unittest.main()
