from unittest.mock import MagicMock, patch

import pytest

from src.server import RAA_TOOLS, call_tool


def test_take_nap_tool_registration():
    """Verify take_nap is in the tool list."""
    tool_names = [t.name for t in RAA_TOOLS]
    assert "run_sleep_cycle" in tool_names

@pytest.mark.asyncio
async def test_run_sleep_cycle_execution():
    """Verify run_sleep_cycle executes SleepCycle.dream."""
    with patch("src.server.get_raa_context") as mock_get_raa:
        # Setup mocks
        mock_ctx = MagicMock()
        mock_sleep_cycle = MagicMock()
        mock_ctx.sleep_cycle = mock_sleep_cycle
        mock_get_raa.return_value = mock_ctx

        # Mock dream return value (must be serializable)
        mock_sleep_cycle.dream.return_value = {"sleep_cycle_results": [{"replay": "done", "crystallization": "done"}]}

        # Call tool
        result = await call_tool("run_sleep_cycle", {"epochs": 2})

        # Verify
        mock_sleep_cycle.dream.assert_called_with(2)
        assert "done" in result[0].text

if __name__ == "__main__":
    pytest.main([__file__])
