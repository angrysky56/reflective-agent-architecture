from typing import List, Optional, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD


class ContinuityField:
    """
    Implements the Identity Manifold as a set of Anchor States stored in a vector database.
    Calculates drift by projecting query vectors onto the local tangent space of nearest anchors.
    """

    def __init__(self, embedding_dim: int, k_neighbors: int = 5):
        self.embedding_dim = embedding_dim
        self.k_neighbors = k_neighbors
        self.anchors: List[np.ndarray] = []
        # In a real system, this would be a FAISS index or similar
        self._anchor_matrix: Optional[np.ndarray] = None

    def add_anchor(self, vector: np.ndarray) -> None:
        """
        Adds a state vector to the manifold (Identity).
        vector: numpy array of shape (embedding_dim,)
        """
        if vector.shape != (self.embedding_dim,):
            raise ValueError(f"Vector dimension mismatch. Expected {self.embedding_dim}, got {vector.shape}")

        self.anchors.append(vector)
        # Inefficient for large scale, but fine for prototype
        self._anchor_matrix = np.array(self.anchors)

    def compute_projection(self, query_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Projects the query_vector onto the local tangent space of the k-nearest anchors.
        Returns:
            projection: The vector projected onto the manifold.
            residual: The vector difference (drift).
        """
        if not self.anchors:
            raise ValueError("Continuity Field is empty. Add anchors first.")

        if self._anchor_matrix is None:
             self._anchor_matrix = np.array(self.anchors)

        if len(self.anchors) < self.k_neighbors:
            # Fallback if not enough anchors
            neighbors = self._anchor_matrix
        else:
            # 1. Find k-nearest neighbors (Euclidean distance)
            dists = np.linalg.norm(self._anchor_matrix - query_vector, axis=1)
            nearest_indices = np.argsort(dists)[: self.k_neighbors]
            neighbors = self._anchor_matrix[nearest_indices]

        # 2. Compute Local Tangent Space (PCA/SVD)
        # Center the neighbors around their mean
        center = np.mean(neighbors, axis=0)
        centered_neighbors = neighbors - center

        # SVD to find principal components (tangent basis)
        # We want to capture the variance of the local neighborhood
        # For k neighbors, we have at most k-1 degrees of freedom
        n_components = min(neighbors.shape[0] - 1, self.embedding_dim)
        if n_components == 0:
            # Single point or identical points -> 0-dim manifold (point)
            return center, query_vector - center

        svd = TruncatedSVD(n_components=n_components)
        svd.fit(centered_neighbors)
        components = svd.components_  # Shape (n_components, embedding_dim)

        # 3. Project query vector onto this space
        # Project (query - center) onto basis, then reconstruct and add center
        query_centered = query_vector - center
        # Dot product with basis vectors to get coefficients
        coeffs = np.dot(query_centered, components.T)
        # Reconstruct projected vector
        projected_centered = np.dot(coeffs, components)
        projection = projected_centered + center

        residual = query_vector - projection

        return projection, residual

    def get_drift_metric(self, query_vector: np.ndarray) -> float:
        """
        Returns the norm of the residual vector (distance from manifold).
        """
        _, residual = self.compute_projection(query_vector)
        return float(np.linalg.norm(residual))
