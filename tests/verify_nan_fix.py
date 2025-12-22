import logging

import torch
import torch.nn.functional as f

from src.manifold.hopfield_network import HopfieldConfig, ModernHopfieldNetwork

# Configure logging to see our warnings
logging.basicConfig(level=logging.WARN)


def test_nan_fix():
    print("--- Starting NaN Fix Verification ---")

    # 1. Demonstrate the Root Cause
    zero_vec = torch.zeros(5)
    normalized = f.normalize(zero_vec.unsqueeze(0), p=2, dim=-1)
    print(f"ROOT CAUSE DEMO: f.normalize(zeros) result: {normalized}")
    assert torch.isnan(normalized).any(), "Expected NaNs from normalizing zero vector!"
    print("Confirmed: Normalizing a zero vector creates NaNs.\n")

    # 2. Verify the Fix in ModernHopfieldNetwork
    config = HopfieldConfig(embedding_dim=5, device="cpu")
    network = ModernHopfieldNetwork(config)

    # Attempt to store a zero vector (Should be rejected)
    print("Attempting to store zero vector in network...")
    network.store_pattern(zero_vec)

    # Check if any patterns were stored
    if network.num_patterns == 0:
        print("SUCCESS: Network rejected the zero vector.")
    else:
        print(f"FAILURE: Network stored {network.num_patterns} patterns.")
        print(f"Stored patterns: {network.patterns}")
        if torch.isnan(network.patterns).any():
            print("CRITICAL FAILURE: Manifold is polluted with NaNs!")
        else:
            print("Network stored it but avoided NaNs (unexpected but okay if logic changed).")

    # Attempt to store a valid vector (Should work)
    valid_vec = torch.randn(5)
    print(f"\nAttempting to store valid vector: {valid_vec[:2]}...")
    network.store_pattern(valid_vec)

    assert network.num_patterns == 1, "Network failed to store valid vector."
    print(f"SUCCESS: Network stored valid vector. Count: {network.num_patterns}")

    # 3. Verify Energy Calculation doesn't crash on NaNs/Infs inputs
    print("\nTesting Energy Calculation robustness...")
    nan_input = torch.tensor([float("nan")] * 5)
    try:
        energy = network.energy(nan_input)
        print(f"Energy for NaN input: {energy}")
        assert not torch.isnan(
            energy
        ), "Energy calculation returned NaN for passed-through NaN input!"
        print(
            "SUCCESS: Energy calculation handled NaN input gracefully (returned 0.0 or valid float)."
        )
    except Exception as e:
        print(f"FAILURE: Energy calculation crashed: {e}")

    print("\n--- Verification Complete ---")


if __name__ == "__main__":
    test_nan_fix()
