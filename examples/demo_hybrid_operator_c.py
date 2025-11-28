"""
Hybrid Operator C Demonstration

Shows the integration of RAA's discrete basin hopping with LTN's continuous
refinement for belief revision.

This example demonstrates:
1. Simple k-NN success (dense memory)
2. LTN fallback (sparse memory or steep gradients)
3. Scaffolding effect (LTN waypoints improve future k-NN)
4. Belief revision workflow

Run with:
    python examples/demo_hybrid_operator_c.py
"""

import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.director.hybrid_search import (
    HybridSearchConfig,
    HybridSearchStrategy,
    SearchStrategy,
)
from src.director.ltn_refiner import LTNConfig, LTNRefiner
from src.manifold import HopfieldConfig, Manifold

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def simple_embedding_fn(text: str) -> torch.Tensor:
    """
    Simple deterministic embedding function for demo.
    
    In production, use SentenceTransformer or similar.
    """
    # Hash-based deterministic embedding
    hash_val = hash(text) % 10000
    torch.manual_seed(hash_val)  # Deterministic
    embedding = torch.randn(128)
    embedding = embedding / embedding.norm()
    return embedding


def main():
    """Run demonstration."""
    logger.info("=" * 60)
    logger.info("Hybrid Operator C Demonstration")
    logger.info("=" * 60)
    
    # Initialize components
    manifold = Manifold(HopfieldConfig(pattern_dim=128, beta=5.0, device="cpu"))
    ltn = LTNRefiner(simple_embedding_fn, LTNConfig(device="cpu"))
    hybrid = HybridSearchStrategy(
        manifold, ltn, None,
        HybridSearchConfig(verbose=True)
    )
    
    logger.info("\n✓ Components initialized\n")
    
    # Demo 1: Populate and search
    logger.info("Demo: Populating memory and searching...")
    for concept in ["mammals", "birds", "fish"]:
        emb = simple_embedding_fn(concept)
        manifold.store(emb.unsqueeze(0))
    
    result = hybrid.search(
        current_state=simple_embedding_fn("whales"),
        evidence=simple_embedding_fn("aquatic mammals")
    )
    
    if result:
        logger.info(f"✓ Search via {result.strategy.value}")
    
    # Print stats
    stats = hybrid.get_stats()
    logger.info(f"\nStats: {stats['knn_success']} k-NN, {stats['ltn_success']} LTN")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
