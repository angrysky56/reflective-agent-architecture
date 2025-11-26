"""
Tests for Matrix Monitor (Cognitive Proprioception).
"""

import pytest
import torch

from src.director.matrix_monitor import MatrixMonitor, MatrixMonitorConfig


@pytest.fixture
def monitor():
    config = MatrixMonitorConfig(
        num_heads=4,
        embedding_dim=64,
        thumbnail_size=4,
        beta=10.0,
        device="cpu"
    )
    return MatrixMonitor(config)

def test_process_attention_shape(monitor):
    """Test that attention processing returns correct embedding shape."""
    # Batch=2, Heads=4, Seq=10, Seq=10
    attention = torch.rand(2, 4, 10, 10)
    embedding = monitor.process_attention(attention)

    assert embedding.shape == (2, 64)
    # Check normalization
    norm = torch.norm(embedding, p=2, dim=-1)
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)

def test_register_and_check_state(monitor):
    """Test learning and recognizing a state."""
    # Create a "Focused" pattern (diagonal)
    focused_attn = torch.eye(10).unsqueeze(0).unsqueeze(0).repeat(1, 4, 1, 1) # (1, 4, 10, 10)

    # Register it
    monitor.register_state(focused_attn, "Focused")

    # Check it
    label, energy = monitor.check_state(focused_attn)
    assert label == "Focused"
    assert energy < 0  # Should be low energy (stable)

    # Check a different pattern (random)
    random_attn = torch.rand(1, 4, 10, 10)
    label_rand, energy_rand = monitor.check_state(random_attn)

    # Energy should be higher for unknown pattern
    assert energy_rand > energy

def test_visualize_topology(monitor):
    """Smoke test for visualization."""
    attention = torch.rand(1, 4, 10, 10)
    viz = monitor.visualize_topology(attention)
    assert isinstance(viz, str)
    assert "Cognitive Topology" in viz
