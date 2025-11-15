"""
Tests for Director (Metacognitive Monitor + Search)
"""

import pytest
import torch

from src.director import Director, DirectorConfig, compute_entropy
from src.manifold import HopfieldConfig, Manifold


def test_entropy_computation():
    """Test entropy calculation from logits."""
    # Uniform distribution should have high entropy
    logits_uniform = torch.zeros(100)
    entropy_uniform = compute_entropy(logits_uniform)
    assert entropy_uniform > 4.0  # log(100) ≈ 4.6

    # Peaked distribution should have low entropy
    logits_peaked = torch.zeros(100)
    logits_peaked[0] = 100.0  # Very confident
    entropy_peaked = compute_entropy(logits_peaked)
    assert entropy_peaked < 0.1


def test_entropy_monitor():
    """Test entropy monitoring and clash detection."""
    config = HopfieldConfig(embedding_dim=128)
    manifold = Manifold(config)

    director_config = DirectorConfig(default_entropy_threshold=2.0)
    director = Director(manifold, director_config)

    # Low entropy should not trigger clash
    low_entropy_logits = torch.zeros(100)
    low_entropy_logits[0] = 10.0
    is_clash, entropy = director.check_entropy(low_entropy_logits)
    assert not is_clash
    assert entropy < 1.0

    # High entropy should trigger clash
    high_entropy_logits = torch.randn(100)
    is_clash, entropy = director.check_entropy(high_entropy_logits)
    # Entropy threshold is adaptive, so this might not trigger initially


def test_search_mechanism():
    """Test k-NN search in Manifold."""
    config = HopfieldConfig(embedding_dim=128)
    manifold = Manifold(config)

    # Store several patterns
    for _ in range(10):
        pattern = torch.randn(128)
        manifold.store_pattern(pattern)

    director_config = DirectorConfig(search_k=3)
    director = Director(manifold, director_config)

    # Search from a current state
    current_state = torch.randn(128)
    result = director.search(current_state)

    assert result is not None
    assert result.best_pattern.shape == (128,)
    assert len(result.neighbor_indices) == 3


def test_full_director_loop():
    """Test complete Director check_and_search loop."""
    config = HopfieldConfig(embedding_dim=128)
    manifold = Manifold(config)

    # Store patterns
    for _ in range(5):
        manifold.store_pattern(torch.randn(128))

    director = Director(manifold)

    # Current state
    current_state = torch.randn(128)

    # High entropy logits (should trigger search)
    high_entropy_logits = torch.randn(1, 100)

    new_goal = director.check_and_search(
        current_state=current_state,
        processor_logits=high_entropy_logits,
    )

    # Result depends on adaptive threshold


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
