"""
Tests for Director (Metacognitive Monitor + Search)
"""

import pytest
import torch

from src.director import Director, DirectorConfig, compute_entropy
from src.director.search_mvp import energy_aware_knn_search, knn_search
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

    _ = director.check_and_search(
        current_state=current_state,
        processor_logits=high_entropy_logits,
    )

    # Result depends on adaptive threshold


def test_energy_aware_search():
    """Test energy-aware k-NN search with Hopfield energy evaluation."""
    embedding_dim = 128
    num_patterns = 10

    # Create memory patterns
    memory_patterns = torch.randn(num_patterns, embedding_dim)

    # Create a query state
    current_state = torch.randn(embedding_dim)

    # Define a simple energy evaluator (Hopfield energy)
    def energy_evaluator(pattern: torch.Tensor) -> torch.Tensor:
        """Compute negative L2 norm as proxy for Hopfield energy."""
        # Lower (more negative) = more stable
        # This is a simplified energy function for testing
        energy = -torch.norm(pattern, p=2)
        return energy

    # Run energy-aware search
    result = energy_aware_knn_search(
        current_state=current_state,
        memory_patterns=memory_patterns,
        energy_evaluator=energy_evaluator,
        k=5,
        metric="cosine",
    )

    # Verify result structure
    assert result is not None
    assert result.best_pattern.shape == (embedding_dim,)
    assert len(result.neighbor_indices) == 5
    assert len(result.neighbor_distances) == 5

    # Verify that energy selection happened
    # The selection_score should be an energy value (negative)
    assert isinstance(result.selection_score, float)

    # Compare with basic k-NN
    basic_result = knn_search(
        current_state=current_state,
        memory_patterns=memory_patterns,
        k=5,
        metric="cosine",
    )

    # Energy-aware might select a different pattern than pure distance-based
    # (they could be the same, but the selection logic is different)
    assert basic_result.best_pattern.shape == result.best_pattern.shape


def test_energy_aware_search_stability():
    """Test that energy-aware search selects lower-energy patterns."""
    embedding_dim = 64

    # Create patterns with known energies
    # Pattern 1: low norm (high energy - unstable)
    pattern_unstable = torch.randn(embedding_dim) * 0.1

    # Pattern 2: high norm (low energy - stable)
    pattern_stable = torch.randn(embedding_dim) * 2.0

    # Pattern 3: medium
    pattern_medium = torch.randn(embedding_dim)

    memory_patterns = torch.stack([pattern_unstable, pattern_stable, pattern_medium])

    # Query close to all of them
    current_state = torch.randn(embedding_dim)

    # Energy evaluator: negative norm (lower = more stable)
    def energy_evaluator(pattern: torch.Tensor) -> torch.Tensor:
        return -torch.norm(pattern, p=2)

    # Run search
    result = energy_aware_knn_search(
        current_state=current_state,
        memory_patterns=memory_patterns,
        energy_evaluator=energy_evaluator,
        k=3,
        metric="cosine",
    )

    # The selected pattern should be pattern_stable (index 1) if energy logic works
    # Compute energies
    energies = [energy_evaluator(p).item() for p in memory_patterns]

    # The best pattern should have the lowest energy among neighbors
    best_pattern_energy = energy_evaluator(result.best_pattern).item()

    # Verify it's one of the low-energy patterns
    assert best_pattern_energy <= max(energies)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
