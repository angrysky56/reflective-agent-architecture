"""
Tests for Processor Monitoring Integration.

Verifies that the refactored TransformerDecoder correctly calls
the Director's monitoring method with attention weights.
"""

from unittest.mock import MagicMock

import pytest
import torch

from src.processor.transformer_decoder import ProcessorConfig, TransformerDecoder


@pytest.fixture
def mock_director():
    director = MagicMock()
    return director

@pytest.fixture
def processor(mock_director):
    config = ProcessorConfig(
        vocab_size=100,
        embedding_dim=32,
        num_layers=2,
        num_heads=4,
        max_seq_length=20,
        device="cpu"
    )
    return TransformerDecoder(config, director=mock_director)

def test_monitoring_hook_called(processor, mock_director):
    """Test that monitor_thought_process is called during forward pass."""
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 100, (batch_size, seq_len))

    # Run forward pass
    processor(input_ids)

    # Verify Director was called
    mock_director.monitor_thought_process.assert_called_once()

    # Verify arguments
    args = mock_director.monitor_thought_process.call_args
    attention_weights = args[0][0]

    # Check shape: (batch, num_heads, seq_len, seq_len)
    assert attention_weights.shape == (batch_size, 4, seq_len, seq_len)

    # Check values are valid probabilities
    # Sum over last dimension should be 1.0 (approx)
    sums = attention_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

def test_monitoring_without_director():
    """Test that processor works fine without a director."""
    config = ProcessorConfig(
        vocab_size=100,
        embedding_dim=32,
        num_layers=2,
        num_heads=4,
        device="cpu"
    )
    processor = TransformerDecoder(config, director=None)

    input_ids = torch.randint(0, 100, (2, 10))
    logits, _ = processor(input_ids)

    assert logits.shape == (2, 10, 100)
