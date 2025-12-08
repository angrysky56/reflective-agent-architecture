import torch

from src.manifold import Manifold
from src.manifold.hopfield_network import HopfieldConfig


def reproduce():
    print("reproducing...")
    config = HopfieldConfig(
        embedding_dim=384,
        beta=10.0,
        adaptive_beta=True,
        beta_min=5.0,
        beta_max=50.0,
        device="cpu",
    )
    manifold = Manifold(config)
    print("Manifold initialized.")

    vec = torch.randn(384)
    print("Storing pattern 'state'...")
    manifold.store_pattern(vec, domain="state")
    print("Success state.")

    print("Storing pattern 'agent'...")
    manifold.store_pattern(vec, domain="agent")
    print("Success agent.")

    print("Storing pattern 'action'...")
    manifold.store_pattern(vec, domain="action")
    print("Success action.")

if __name__ == "__main__":
    try:
        reproduce()
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
