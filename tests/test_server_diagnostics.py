"""
Tests for Server Diagnostics Tool.

Verifies that the `diagnose_pointer` tool correctly extracts weights
and returns valid sheaf diagnostics.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.director import CohomologyResult, HodgeDecomposition, MonodromyAnalysis, SheafDiagnostics
from src.server import call_tool, get_raa_context


@pytest.fixture
def mock_raa_context():
    """Mock RAA context with Pointer and Director."""
    # Mock Pointer with RNN
    pointer = MagicMock()
    pointer.rnn = torch.nn.GRU(input_size=16, hidden_size=16, num_layers=1)

    # Mock Director
    director = MagicMock()

    # Mock Diagnosis Result
    mock_diagnosis = SheafDiagnostics(
        cohomology=CohomologyResult(
            h0_dimension=0,
            h1_dimension=1,  # Simulate obstruction
            can_fully_resolve=False,
            singular_values=torch.tensor([1.0, 0.1]),
            null_space_basis=None
        ),
        hodge=HodgeDecomposition(
            harmonic_projector=torch.eye(16),
            diffusive_operator=torch.eye(16),
            harmonic_residual=None,
            diffusive_activation=None,
            eliminable_error=None
        ),
        monodromy=None,
        harmonic_diffusive_overlap=0.05,
        learning_can_proceed=False,
        escalation_recommended=True,
        diagnostic_messages=["Test warning"]
    )
    director.diagnose.return_value = mock_diagnosis

    context = {
        "pointer": pointer,
        "director": director,
        "bridge": MagicMock(),
        "manifold": MagicMock()
    }
    return context

@pytest.mark.asyncio
async def test_diagnose_pointer_tool(mock_raa_context):
    """Test diagnose_pointer tool execution."""

    # Patch get_raa_context to return our mock
    with patch("src.server.get_raa_context", return_value=mock_raa_context):
        # Call the tool
        result = await call_tool("diagnose_pointer", {})

        # Verify result format
        assert len(result) == 1
        import json
        data = json.loads(result[0].text)

        # Check fields
        assert "h1_dimension" in data
        assert data["h1_dimension"] == 1
        assert "can_resolve" in data
        assert data["can_resolve"] is False
        assert "escalation_recommended" in data
        assert data["escalation_recommended"] is True
        assert "messages" in data
        assert "Test warning" in data["messages"]

        # Verify Director was called with weights
        mock_raa_context["director"].diagnose.assert_called_once()
        args = mock_raa_context["director"].diagnose.call_args
        weights = args[0][0]
        assert len(weights) > 0
        assert isinstance(weights[0], torch.Tensor)

@pytest.mark.asyncio
async def test_diagnose_pointer_no_rnn(mock_raa_context):
    """Test error handling when Pointer has no RNN."""
    # Remove RNN from pointer
    del mock_raa_context["pointer"].rnn

    with patch("src.server.get_raa_context", return_value=mock_raa_context):
        result = await call_tool("diagnose_pointer", {})

        assert "Error" in result[0].text
