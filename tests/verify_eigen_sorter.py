import logging
import os
import sys

import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.integration.eigen_sorter import ContextualEigenSorter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_eigen_sorter():
    logger.info("Starting ContextualEigenSorter Verification...")

    # 1. Setup
    dim = 10
    sorter = ContextualEigenSorter(embedding_dim=dim, n_components=2)

    # Create User Basis (Random vector)
    user_vec = np.random.normal(size=dim)
    user_vec /= np.linalg.norm(user_vec)
    user_basis = np.array([user_vec])

    logger.info("Generated User Basis")

    # 2. Generate Memories
    # Group A: Aligned with User (Signal)
    aligned_memories = []
    for _ in range(10):
        noise = np.random.normal(scale=0.1, size=dim)
        vec = user_vec + noise
        vec /= np.linalg.norm(vec)
        aligned_memories.append(vec)

    # Group B: Orthogonal to User (Noise/Psychosis)
    # Create a vector orthogonal to user_vec
    orth_vec = np.random.normal(size=dim)
    orth_vec -= orth_vec.dot(user_vec) * user_vec  # Remove user component
    orth_vec /= np.linalg.norm(orth_vec)

    orthogonal_memories = []
    for _ in range(10):
        noise = np.random.normal(scale=0.1, size=dim)
        vec = orth_vec + noise
        vec /= np.linalg.norm(vec)
        orthogonal_memories.append(vec)

    all_memories = aligned_memories + orthogonal_memories
    # Indices 0-9 are Aligned, 10-19 are Orthogonal

    # 3. Test Compute Eigenvectors & Drift (Aligned Case)
    logger.info("Testing Aligned Case...")
    basis_aligned, _ = sorter.compute_eigenvectors(aligned_memories)
    drift_aligned = sorter.calculate_drift(basis_aligned, user_basis)
    logger.info(f"Drift (Aligned): {drift_aligned:.4f}")
    assert drift_aligned < 0.2, f"Drift should be low for aligned memories, got {drift_aligned}"

    # 4. Test Compute Eigenvectors & Drift (Orthogonal Case)
    logger.info("Testing Orthogonal Case...")
    basis_orth, _ = sorter.compute_eigenvectors(orthogonal_memories)
    drift_orth = sorter.calculate_drift(basis_orth, user_basis)
    logger.info(f"Drift (Orthogonal): {drift_orth:.4f}")
    assert drift_orth > 0.8, f"Drift should be high for orthogonal memories, got {drift_orth}"

    # 5. Test Synchronization (Re-weighting)
    # When we mix them, the SVD might pick an intermediate or one of them.
    # But synchronize should sort Aligned to the top if we force alignment.

    logger.info("Testing Synchronization (Sorting)...")
    # We pass the mixed batch. The sorter should detect alignment of each vector to user_basis
    # forcing alignment because we expect the 'drift' logic to trigger based on the user basis passed.
    # Actually, synchronize calculates drift of the *batch*.
    # If the batch is mixed, drift might be moderate.
    # Let's force it to prioritize user by setting a LOW threshold.

    sorted_indices = sorter.synchronize(
        all_memories, user_basis, drift_threshold=0.0
    )  # Force user alignment

    top_5 = sorted_indices[:5]
    logger.info(f"Top 5 indices: {top_5}")

    # Expect top 5 to be from the aligned group (0-9)
    for idx in top_5:
        assert idx < 10, f"Index {idx} is from orthogonal group, should be aligned."

    logger.info("Verification Passed!")


if __name__ == "__main__":
    test_eigen_sorter()
