import os
import sys

import torch

# Adjust path to include project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, project_root)

from src.integration.precuneus import PrecuneusIntegrator


def test_precuneus_dimensions():
    print("Initializing PrecuneusIntegrator...")
    dim = 384
    precuneus = PrecuneusIntegrator(dim)

    # Create inputs with MISMATCHED dimensions
    # State: 1D (Correct for synthesis)
    state_vec = torch.randn(dim)

    # Agent: 2D (simulating Pointer returning (1, dim))
    agent_vec = torch.randn(1, dim)

    # Action: 1D
    action_vec = torch.randn(dim)

    vectors = {"state": state_vec, "agent": agent_vec, "action": action_vec}

    energies = {"state": 1.0, "agent": 1.0, "action": 1.0}

    print(
        f"Input dimensions: State={state_vec.shape}, Agent={agent_vec.shape}, Action={action_vec.shape}"
    )

    try:
        print("Attempting forward pass...")
        integrated, info = precuneus(vectors, energies)
        print("Success! Forward pass completed.")
        print(f"Output shape: {integrated.shape}")
        print(f"Info: {info}")

        # Verify output is 1D (since we expect 1D result if we squeezed)
        assert integrated.dim() == 1, "Output should be 1D tensor"
        print("Verdict: PASS")

    except Exception as e:
        print(f"Test Failed with Exception: {e}")
        import traceback

        traceback.print_exc()
        print("Verdict: FAIL")


if __name__ == "__main__":
    test_precuneus_dimensions()
