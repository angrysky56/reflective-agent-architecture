
import logging

import torch

from src.director.director_core import DirectorConfig, DirectorMVP
from src.director.sheaf_diagnostics import create_supervision_target
from src.manifold import HopfieldConfig, Manifold
from src.pointer.goal_controller import GoalController, PointerConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_fix():
    print("Initializing components...")
    device = "cpu"
    embedding_dim = 64 # Small dim for testing

    # 1. Setup Pointer
    pointer_cfg = PointerConfig(
        embedding_dim=embedding_dim,
        hidden_dim=embedding_dim,
        controller_type="gru",
        device=device
    )
    pointer = GoalController(pointer_cfg)

    # Initialize pointer weights (random)
    pointer.reset()

    # 2. Setup Director
    # Need a dummy manifold
    hopfield_cfg = HopfieldConfig(embedding_dim=embedding_dim, device=device)
    manifold = Manifold(hopfield_cfg)

    director = DirectorMVP(manifold)

    # 3. Extract weights (same logic as server.py)
    weights = []
    hh = pointer.rnn.weight_hh_l0.detach()

    # Use Identity Extension to create a free vertex (Hidden State)
    # Model: h_{t-1} --(hh)--> h_internal --(I)--> h_t
    # This allows Sheaf Analysis to see "diffusion" (inference) in the internal state
    weights.append(hh)
    weights.append(torch.eye(hh.shape[0]))

    print(f"Extracted {len(weights)} weight matrices (with Identity extension).")

    # 4. Run Diagnosis WITHOUT target (Old behavior)
    print("\n--- Test 1: Diagnosis WITHOUT target (Old Behavior) ---")
    diag_old = director.diagnose(weights)
    print(f"Overlap: {diag_old.harmonic_diffusive_overlap}")

    # 5. Run Diagnosis WITH target (New Behavior)
    print("\n--- Test 2: Diagnosis WITH target (New Behavior) ---")

    # Target must match total edge dim (192 + 192 = 384)
    total_edge_dim = sum(w.shape[0] for w in weights)

    # Use random target to probe topology
    target_error = torch.randn(total_edge_dim)

    diag_new = director.diagnose(weights, target_error=target_error)
    print(f"Overlap: {diag_new.harmonic_diffusive_overlap}")

    if diag_new.harmonic_diffusive_overlap > 0.0001:
        print("\nSUCCESS: Overlap is non-zero with target!")
    else:
        print("\nFAILURE: Overlap is still zero.")

if __name__ == "__main__":
    verify_fix()
