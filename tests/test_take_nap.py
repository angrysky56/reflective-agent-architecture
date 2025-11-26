from unittest.mock import MagicMock, patch

import pytest

from src.server import RAA_TOOLS, call_tool


def test_take_nap_tool_registration():
    """Verify take_nap is in the tool list."""
    tool_names = [t.name for t in RAA_TOOLS]
    assert "take_nap" in tool_names

@pytest.mark.asyncio
async def test_take_nap_execution():
    """Verify take_nap executes SleepCycle.dream."""
    with patch("src.server.get_workspace") as mock_get_workspace, \
         patch("src.server.get_raa_context") as mock_get_raa, \
         patch("src.server.SleepCycle") as MockSleepCycle:

        # Setup mocks
        mock_workspace = MagicMock()
        mock_get_workspace.return_value = mock_workspace

        mock_sleep_instance = MockSleepCycle.return_value
        mock_sleep_instance.dream.return_value = {"replay": "done", "crystallization": "done"}

        # Call tool
        result = await call_tool("take_nap", {"epochs": 2})

        # Verify
        MockSleepCycle.assert_called_with(workspace=mock_workspace)
        mock_sleep_instance.dream.assert_called_with(epochs=2)
        assert "done" in result[0].text

if __name__ == "__main__":
    pytest.main([__file__])
