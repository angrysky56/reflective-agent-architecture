
import logging
from unittest.mock import MagicMock

import torch

from src.director.hybrid_search import HybridSearchConfig, HybridSearchResult, HybridSearchStrategy, SearchStrategy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_revise_preference():
    """
    Test that HybridSearchStrategy prefers k-NN over LTN by default,
    and verify that we can force LTN.
    """
    print("\n=== Testing Revise Tool Preference ===")

    # 1. Setup Mocks
    mock_manifold = MagicMock()
    mock_ltn = MagicMock()
    mock_sheaf = MagicMock()

    # Mock Manifold behavior
    # Assume we have 10 patterns in memory
    mock_manifold.get_patterns.return_value = torch.randn(10, 768)

    # Mock Energy function (lower is better/more stable)
    mock_manifold.energy.side_effect = lambda x: torch.tensor(-1.0) # Always stable

    # Mock Attention for Sheaf validation
    mock_manifold.get_attention.return_value = torch.tensor([1.0])

    # Mock LTN Refiner
    # Should return a "new" pattern different from inputs
    mock_ltn.refine.return_value = torch.randn(768)
    mock_ltn.config.max_iterations = 10

    # 2. Initialize Strategy
    config = HybridSearchConfig(
        min_memory_size=1, # Ensure k-NN runs
        knn_exclude_threshold=0.0, # Allow any match
        verbose=True
    )
    strategy = HybridSearchStrategy(mock_manifold, mock_ltn, mock_sheaf, config)

    # 3. Define Inputs
    current_state = torch.randn(768)
    evidence = torch.randn(768)

    # 4. Test Case 1: Default Behavior (Should prefer k-NN)
    print("\n--- Test Case 1: Default Behavior ---")
    # We need to mock energy_aware_knn_search to return a success
    # Since it's imported in hybrid_search, we can't easily mock it without patching.
    # Instead, let's rely on the fact that _try_knn_search calls it.
    # But wait, energy_aware_knn_search is a standalone function.
    # Let's patch it in the module.

    from src.director import hybrid_search

    # Create a dummy result for k-NN
    knn_result = HybridSearchResult(
        best_pattern=torch.randn(768),
        neighbor_indices=[0],
        neighbor_distances=torch.tensor([0.1]),
        selection_score=-1.0,
        strategy=SearchStrategy.KNN,
        knn_attempted=True
    )

    # Patch the function
    original_knn = hybrid_search.energy_aware_knn_search
    hybrid_search.energy_aware_knn_search = MagicMock(return_value=knn_result)

    try:
        result = strategy.search(current_state, evidence=evidence)

        print(f"Strategy used: {result.strategy}")
        if result.strategy == SearchStrategy.KNN:
            print("PASS: Default behavior preferred k-NN as expected (reproducing the 'issue').")
        else:
            print(f"FAIL: Expected k-NN, got {result.strategy}")

    finally:
        # Restore
        hybrid_search.energy_aware_knn_search = original_knn

    # 5. Test Case 2: Forcing LTN (The Fix)
    print("\n--- Test Case 2: Forcing LTN ---")

    # Re-patch for this test
    hybrid_search.energy_aware_knn_search = MagicMock(return_value=knn_result)

    try:
        # Try to pass force_ltn=True (This should fail before we implement the fix)
        try:
            result = strategy.search(current_state, evidence=evidence, force_ltn=True)
            print(f"Strategy used: {result.strategy}")

            if result.strategy == SearchStrategy.LTN:
                print("PASS: Successfully forced LTN.")
            else:
                print(f"FAIL: Expected LTN, got {result.strategy}")

        except TypeError:
            print("PASS (Expected): 'force_ltn' argument not yet supported.")
        except Exception as e:
            print(f"ERROR: {e}")

    finally:
        hybrid_search.energy_aware_knn_search = original_knn

if __name__ == "__main__":
    test_revise_preference()
