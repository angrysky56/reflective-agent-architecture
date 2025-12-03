from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from src.server import call_tool
from src.substrate.energy import EnergyDepletionError, EnergyToken, MeasurementCost


@pytest.mark.asyncio
async def test_energy_deduction():
    """Verify energy is deducted for costly operations."""
    with patch("src.server.get_raa_context") as mock_get_raa:
        # Setup mocks
        mock_ctx = MagicMock()
        mock_workspace = MagicMock()
        mock_ledger = MagicMock()

        mock_ctx.workspace = mock_workspace
        mock_workspace.ledger = mock_ledger
        mock_get_raa.return_value = mock_ctx

        # Mock bridge execution to avoid errors
        mock_ctx.get_bridge.return_value.execute_monitored_operation.return_value = {"synthesis": "test"}

        # Call synthesize (Cost: 3.0)
        await call_tool("synthesize", {"node_ids": ["1", "2"]})

        # Verify transaction recorded
        mock_ledger.record_transaction.assert_called_once()
        args, _ = mock_ledger.record_transaction.call_args
        cost = args[0]
        assert cost.energy.amount == Decimal("3.0")
        assert cost.operation_name == "synthesize"

@pytest.mark.asyncio
async def test_energy_recharge():
    """Verify run_sleep_cycle recharges energy."""
    with patch("src.server.get_raa_context") as mock_get_raa:
        # Setup mocks
        mock_ctx = MagicMock()
        mock_workspace = MagicMock()
        mock_ledger = MagicMock()
        mock_sleep_cycle = MagicMock()

        mock_ctx.workspace = mock_workspace
        mock_ctx.sleep_cycle = mock_sleep_cycle
        mock_workspace.ledger = mock_ledger
        mock_get_raa.return_value = mock_ctx

        # Mock dream return
        mock_sleep_cycle.dream.return_value = {"status": "dreaming"}

        # Call run_sleep_cycle
        await call_tool("run_sleep_cycle", {"epochs": 1})

        # Verify recharge called
        mock_ledger.recharge.assert_called_once()

@pytest.mark.asyncio
async def test_energy_depletion():
    """Verify EnergyDepletionError stops execution."""
    # This test requires mocking the ledger to raise the error
    with patch("src.server.get_raa_context") as mock_get_raa:
        mock_ctx = MagicMock()
        mock_workspace = MagicMock()
        mock_ledger = MagicMock()

        mock_ctx.workspace = mock_workspace
        mock_workspace.ledger = mock_ledger
        mock_get_raa.return_value = mock_ctx

        # Mock record_transaction to raise error
        mock_ledger.record_transaction.side_effect = EnergyDepletionError("Low Energy")

        # Call tool
        try:
            await call_tool("evolve_formula", {"data_points": []})
        except EnergyDepletionError:
            pass # Expected, but call_tool might catch it.

        # Actually call_tool catches generic exceptions, let's see if it propagates or returns error text
        # In server.py: except Exception as e: return [TextContent(text=f"Critical Server Error: {str(e)}")]

        result = await call_tool("evolve_formula", {"data_points": []})
        assert "Error: Low Energy" in result[0].text
        assert "Low Energy" in result[0].text
