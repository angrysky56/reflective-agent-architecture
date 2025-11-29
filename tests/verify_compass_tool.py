
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.server import RAAServerContext


class TestCompassTool(unittest.IsolatedAsyncioTestCase):
    async def test_consult_compass(self):
        # Setup context
        ctx = RAAServerContext()
        ctx.workspace = MagicMock()

        # Mock Director and COMPASS
        mock_director = MagicMock()
        mock_compass = MagicMock()
        mock_director.compass = mock_compass

        # Mock process_task to be async
        mock_compass.process_task = AsyncMock(return_value={"status": "success", "result": "Task completed"})

        ctx.raa_context = {
            "director": mock_director,
            "device": "cpu",
            "bridge": MagicMock(),
            "agent_factory": MagicMock()
        }
        ctx.raa_context["agent_factory"].active_agents = []

        # Call tool
        result = await ctx.call_tool("consult_compass", {"task": "Test task", "context": {"foo": "bar"}})

        # Verify
        mock_compass.process_task.assert_called_once_with("Test task", {"foo": "bar"})
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"], "Task completed")
        print("✅ consult_compass tool correctly delegates to COMPASS")

if __name__ == '__main__':
    unittest.main()
