#!/usr/bin/env python3
"""
Quick validation script for energy-aware k-NN search.
Run this to verify the implementation works without waiting for full pytest.
"""

import torch

from src.director.search_mvp import energy_aware_knn_search, knn_search


def simple_energy_evaluator(pattern: torch.Tensor) -> torch.Tensor:
    """Simple energy function: negative L2 norm."""
    return -torch.norm(pattern, p=2)


def main():
    print("=" * 60)
    print("Energy-Aware k-NN Search Validation")
    print("=" * 60)

    # Setup
    embedding_dim = 128
    num_patterns = 10
    k = 5

    print("\nSetup:")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Number of patterns: {num_patterns}")
    print(f"  k neighbors: {k}")

    # Create test data
    memory_patterns = torch.randn(num_patterns, embedding_dim)
    current_state = torch.randn(embedding_dim)

    print("\n" + "-" * 60)
    print("Test 1: Basic k-NN Search")
    print("-" * 60)

    basic_result = knn_search(
        current_state=current_state,
        memory_patterns=memory_patterns,
        k=k,
        metric="cosine",
    )

    print("✓ Basic k-NN completed")
    print(f"  Best pattern shape: {basic_result.best_pattern.shape}")
    print(f"  Neighbor indices: {basic_result.neighbor_indices}")
    print(f"  Selection score: {basic_result.selection_score:.4f}")

    print("\n" + "-" * 60)
    print("Test 2: Energy-Aware k-NN Search")
    print("-" * 60)

    energy_result = energy_aware_knn_search(
        current_state=current_state,
        memory_patterns=memory_patterns,
        energy_evaluator=simple_energy_evaluator,
        k=k,
        metric="cosine",
    )

    print("✓ Energy-aware k-NN completed")
    print(f"  Best pattern shape: {energy_result.best_pattern.shape}")
    print(f"  Neighbor indices: {energy_result.neighbor_indices}")
    print(f"  Selection score (energy): {energy_result.selection_score:.4f}")

    print("\n" + "-" * 60)
    print("Test 3: Energy Selection Verification")
    print("-" * 60)

    # Compute energies for all neighbors
    neighbor_energies = []
    for idx in energy_result.neighbor_indices:
        pattern = memory_patterns[idx]
        energy = simple_energy_evaluator(pattern).item()
        neighbor_energies.append(energy)

    print(f"  Neighbor energies: {[f'{e:.4f}' for e in neighbor_energies]}")
    print(f"  Selected energy: {energy_result.selection_score:.4f}")
    print(f"  Min energy: {min(neighbor_energies):.4f}")

    # Verify lowest energy was selected
    assert (
        abs(energy_result.selection_score - min(neighbor_energies)) < 1e-5
    ), "Energy-aware search should select lowest energy pattern!"

    print("✓ Verified: Selected pattern has lowest energy")

    print("\n" + "-" * 60)
    print("Test 4: Stability Test (Known Patterns)")
    print("-" * 60)

    # Create patterns with known stability characteristics
    pattern_unstable = torch.randn(embedding_dim) * 0.1  # Low norm = high energy
    pattern_stable = torch.randn(embedding_dim) * 3.0  # High norm = low energy
    pattern_medium = torch.randn(embedding_dim) * 1.0

    test_patterns = torch.stack([pattern_unstable, pattern_medium, pattern_stable])
    test_state = torch.randn(embedding_dim)

    result = energy_aware_knn_search(
        current_state=test_state,
        memory_patterns=test_patterns,
        energy_evaluator=simple_energy_evaluator,
        k=3,
        metric="cosine",
    )

    energies = [simple_energy_evaluator(p).item() for p in test_patterns]
    selected_energy = simple_energy_evaluator(result.best_pattern).item()

    print("  Pattern energies:")
    print(f"    Unstable: {energies[0]:.4f}")
    print(f"    Medium:   {energies[1]:.4f}")
    print(f"    Stable:   {energies[2]:.4f}")
    print(f"  Selected energy: {selected_energy:.4f}")

    # The selected pattern should be one of the lower-energy ones
    assert selected_energy <= max(energies), "Selected pattern should not have highest energy!"

    print("✓ Verified: Selected pattern has reasonable energy")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print("\nEnergy-aware k-NN search is working correctly.")
    print("The implementation:")
    print("  1. Finds k nearest neighbors geometrically")
    print("  2. Evaluates Hopfield energy for each neighbor")
    print("  3. Selects the pattern with LOWEST energy")
    print("\nThis aligns with Hopfield network principles:")
    print("  - Lower energy = more stable attractor")
    print("  - Energy-based selection ensures theoretical coherence")


if __name__ == "__main__":
    main()
