from unittest.mock import MagicMock, patch

import pytest

from src.server import call_tool


@pytest.mark.asyncio
async def test_consult_curiosity_tool():
    """Verify consult_curiosity tool calls curiosity module."""
    with patch("src.server.get_raa_context") as mock_get_raa:
        # Setup mocks
        mock_ctx = MagicMock()
        mock_workspace = MagicMock()
        mock_curiosity = MagicMock()

        mock_ctx.workspace = mock_workspace
        mock_workspace.curiosity = mock_curiosity
        mock_get_raa.return_value = mock_ctx

        # Case 1: Goal proposed
        mock_curiosity.propose_goal.return_value = "Investigate X"
        result = await call_tool("consult_curiosity", {})
        assert "Investigate X" in result[0].text
        assert "success" in result[0].text

        # Case 2: No goal (not bored)
        mock_curiosity.propose_goal.return_value = None
        result = await call_tool("consult_curiosity", {})
        assert "No boredom-driven goals" in result[0].text
        assert "idle" in result[0].text
