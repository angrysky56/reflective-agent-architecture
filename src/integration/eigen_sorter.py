import logging
from typing import List, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class ContextualEigenSorter:
    """
    The 'Special DB' mechanism for Topological Simplification.

    Implements:
    1. Global SVD (Singular Value Decomposition) on the Experience Buffer.
    2. 'Sanity Check': Calculates drift between Agent's Eigen-Basis and User's Manifold.
    3. Re-weighting: Sorts memories by their projection onto the 'Sanity Axis'.
    """

    def __init__(self, embedding_dim: int = 1536, n_components: int = 5):
        self.embedding_dim = embedding_dim
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components)

    def compute_eigenvectors(self, memories: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs SVD on the memory matrix to find Principal Components (Concepts).
        Returns: (components, singular_values)
        """
        if not memories:
            return np.zeros((self.n_components, self.embedding_dim)), np.zeros(self.n_components)

        matrix = np.array(memories)
        if matrix.shape[0] < self.n_components:
            # Not enough data for full SVD, start with what we have
            n = matrix.shape[0]
            svd = TruncatedSVD(n_components=n)
            svd.fit(matrix)
            return svd.components_, svd.singular_values_

        self.svd.fit(matrix)
        return self.svd.components_, self.svd.singular_values_

    def calculate_drift(self, agent_basis: np.ndarray, user_basis: np.ndarray) -> float:
        """
        Calculates the 'Sanity Drift' (Topological Fracture) between Agent and User.
        Metric: Principal Angles between subspaces (or simple Cosine Similarity of top component).
        Returns: Drift score [0.0, 1.0], where 0.0 is perfect alignment, 1.0 is orthogonal.
        """
        if agent_basis.shape[0] == 0 or user_basis.shape[0] == 0:
            return 0.0  # Default to aligned if no history

        # Simplification: Compare primary eigenvectors (Dominant Concept)
        primary_agent = agent_basis[0]
        primary_user = user_basis[0]

        similarity = float(cosine_similarity([primary_agent], [primary_user])[0][0])
        drift = 1.0 - abs(similarity)  # Abs because eigen-direction sign is arbitrary

        return drift

    def synchronize(
        self, memories: List[np.ndarray], user_basis: np.ndarray, drift_threshold: float = 0.3
    ) -> List[int]:
        """
        Sorts indices of memories based on alignment with the 'Sanity Axis' (User Basis).
        If drift is high, it prioritizes User Alignment.
        If drift is low, it prioritizes Internal Coherence (Agent Basis).
        """
        if not memories:
            return []

        memory_matrix = np.array(memories)
        agent_basis, _ = self.compute_eigenvectors(memories)

        drift = self.calculate_drift(agent_basis, user_basis)
        logger.info(f"Contextual Eigen-Sorter: Sanity Drift = {drift:.4f}")

        # Define the 'Sanity Axis' for sorting
        if drift > drift_threshold:
            logger.warning(
                "High Drift detected! Forcing Topological Realignment (Dreaming of User)."
            )
            # Project onto User's Principal Component
            target_axis = user_basis[0]
        else:
            logger.info("Drift nominal. Optimizing for Internal Coherence.")
            # Project onto Agent's OWN Principal Component
            target_axis = agent_basis[0]

        # Calculate projection scores: |v . axis|
        scores = np.abs(np.dot(memory_matrix, target_axis))

        # Return indices sorted by score (descending)
        sorted_indices = np.argsort(scores)[::-1]

        return [int(i) for i in sorted_indices.tolist()]
