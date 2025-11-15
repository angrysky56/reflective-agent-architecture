"""
Tests for Manifold (Modern Hopfield Network)
"""

import pytest
import torch

from src.manifold import Manifold, HopfieldConfig


def test_manifold_initialization():
    """Test that Manifold initializes correctly."""
    config = HopfieldConfig(embedding_dim=128)
    manifold = Manifold(config)

    assert manifold.embedding_dim == 128
    assert manifold.num_patterns == 0


def test_pattern_storage():
    """Test storing and retrieving patterns."""
    config = HopfieldConfig(embedding_dim=128)
    manifold = Manifold(config)

    # Store a pattern
    pattern = torch.randn(128)
    manifold.store_pattern(pattern)

    assert manifold.num_patterns == 1

    # Store multiple patterns
    patterns = torch.randn(5, 128)
    for p in patterns:
        manifold.store_pattern(p)

    assert manifold.num_patterns == 6


def test_energy_computation():
    """Test Hopfield energy computation."""
    config = HopfieldConfig(embedding_dim=128, beta=1.0)
    manifold = Manifold(config)

    # Store patterns
    patterns = torch.randn(3, 128)
    for p in patterns:
        manifold.store_pattern(p)

    # Compute energy
    state = torch.randn(128)
    energy = manifold.energy(state)

    assert isinstance(energy, torch.Tensor)
    assert energy.dim() == 0  # Scalar


def test_retrieval():
    """Test associative retrieval."""
    config = HopfieldConfig(embedding_dim=128, update_steps=10)
    manifold = Manifold(config)

    # Store patterns
    pattern1 = torch.randn(128)
    pattern2 = torch.randn(128)
    manifold.store_pattern(pattern1)
    manifold.store_pattern(pattern2)

    # Query with noisy version of pattern1
    query = pattern1 + 0.1 * torch.randn(128)
    retrieved, energy_traj = manifold.retrieve(query)

    assert retrieved.shape == (128,)
    assert energy_traj.shape[0] == 11  # update_steps + 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
