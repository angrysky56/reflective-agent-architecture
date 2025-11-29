
import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import torch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.compass.compass_framework import COMPASS
from src.director.director_core import DirectorConfig, DirectorMVP


class TestCompassAutomation(unittest.IsolatedAsyncioTestCase):
    async def test_automation_trigger(self):
        # Setup mocks
        mock_manifold = MagicMock()
        mock_manifold.beta = 10.0

        config = DirectorConfig(device="cpu")

        # Mock oMCD controller to return high allocation
        mock_omcd = MagicMock()
        mock_omcd.determine_resource_allocation.return_value = {
            "amount": 90.0, # High allocation > 80.0
            "confidence": 0.1,
            "net_benefit": 10.0
        }

        # Mock COMPASS
        mock_compass = MagicMock()
        mock_compass.omcd_controller = mock_omcd
        mock_compass.process_task = AsyncMock(return_value={"status": "intervention_complete"})

        # Initialize Director
        director = DirectorMVP(manifold=mock_manifold, config=config)
        director.compass = mock_compass

        # Mock monitor to return high entropy
        director.monitor = MagicMock()
        director.monitor.check_logits.return_value = (True, 5.0) # High entropy
        director.monitor.get_threshold.return_value = 2.0

        # Mock search to avoid errors
        director.hybrid_search = MagicMock()
        director.hybrid_search.search.return_value = None

        # Run check_and_search
        # We need to run this in a way that allows the async task to be scheduled and run
        # check_and_search is sync, but it schedules a task on the running loop

        current_state = torch.randn(10)
        logits = torch.randn(1, 10)

        # We need to ensure there is a running loop, which IsolatedAsyncioTestCase provides
        director.check_and_search(current_state, logits)

        # Allow async tasks to run
        await asyncio.sleep(0.1)

        # Verify process_task was called
        mock_compass.process_task.assert_called()
        call_args = mock_compass.process_task.call_args
        self.assertIn("High Entropy Intervention", call_args[0][0])
        print("✅ COMPASS intervention triggered automatically on high entropy")

if __name__ == '__main__':
    unittest.main()
