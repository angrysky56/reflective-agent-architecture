import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from neo4j import GraphDatabase
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)


class ContinuityField:
    """
    Implements the Identity Manifold as a Fiber Bundle (E, B, pi, F) per TKUI Formalization.

    Components:
    - Base Space (B): Temporal trajectory of the agent.
    - Fiber (F): State space manifold at time t.
    - Connection (Delta t): Temporal transport ensuring coherence.
    - Section (sigma): Spatial coherence constraint.

    Features:
    - Causal Impact Signatures: Integrated influence on environment.
    - Self-Model Invariants: Parameters resistant to updates.
    - Transformation History: Weighted sum of past modifications.
    """

    def __init__(
        self,
        embedding_dim: int,
        k_neighbors: int = 5,
        neo4j_uri: Optional[str] = None,
        neo4j_auth: Optional[Tuple[str, str]] = None,
    ):
        self.embedding_dim = embedding_dim
        self.k_neighbors = k_neighbors
        self.anchors: List[np.ndarray] = []
        self._anchor_matrix: Optional[np.ndarray] = None

        # TKUI Parameters
        self.temporal_connection_strength = 1.0  # Delta t weight
        self.spatial_coherence_threshold = 0.7  # Sigma threshold

        # Neo4j Integration
        self.driver = None
        if neo4j_uri:
            try:
                self.driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
                logger.info("ContinuityField connected to Neo4j.")
            except Exception as e:
                logger.warning(f"Failed to connect to Neo4j: {e}")

    def close(self) -> None:
        if self.driver:
            self.driver.close()

    def add_anchor(self, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Adds a state vector to the manifold (Identity).
        """
        if vector.shape != (self.embedding_dim,):
            raise ValueError(
                f"Vector dimension mismatch. Expected {self.embedding_dim}, got {vector.shape}"
            )

        self.anchors.append(vector)
        self._anchor_matrix = np.array(self.anchors)

        if self.driver and metadata:
            self._persist_anchor(vector, metadata)

    def _persist_anchor(self, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Persist anchor to Neo4j as part of the Continuity Field."""
        if self.driver is None:
            logger.warning("Cannot persist anchor: Neo4j driver not initialized")
            return
        with self.driver.session() as session:
            session.run(
                """
                CREATE (c:ContinuityField {
                    timestamp: $timestamp,
                    embedding: $embedding,
                    agent_id: $agent_id,
                    causal_signature: $causal_signature
                })
            """,
                {
                    "timestamp": time.time(),
                    "embedding": vector.tolist(),
                    "agent_id": metadata.get("agent_id", "unknown"),
                    "causal_signature": metadata.get("causal_signature", 0.0),
                },
            )

    def compute_projection(self, query_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Projects the query_vector onto the local tangent space of the k-nearest anchors.
        """
        if not self.anchors:
            # If empty, return vector itself and zero residual (or handle as error)
            # For initialization robustness, we can treat empty field as perfect match (identity not yet formed)
            return query_vector, np.zeros_like(query_vector)

        if self._anchor_matrix is None:
            self._anchor_matrix = np.array(self.anchors)

        if len(self.anchors) < self.k_neighbors:
            neighbors = self._anchor_matrix
        else:
            dists = np.linalg.norm(self._anchor_matrix - query_vector, axis=1)
            nearest_indices = np.argsort(dists)[: self.k_neighbors]
            neighbors = self._anchor_matrix[nearest_indices]

        # Compute Local Tangent Space (PCA/SVD)
        center = np.mean(neighbors, axis=0)
        centered_neighbors = neighbors - center

        n_components = min(neighbors.shape[0] - 1, self.embedding_dim)
        if n_components == 0:
            return center, query_vector - center

        svd = TruncatedSVD(n_components=n_components)
        svd.fit(centered_neighbors)
        components = svd.components_

        query_centered = query_vector - center
        coeffs = np.dot(query_centered, components.T)
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

    def validate_coherence(self, current_state: np.ndarray, proposed_state: np.ndarray) -> float:
        """
        Validates if a proposed state maintains topological coherence with the field.
        Returns a coherence score [0, 1].
        """
        # 1. Check drift from manifold
        drift_current = self.get_drift_metric(current_state)
        drift_proposed = self.get_drift_metric(proposed_state)

        # 2. Check local smoothness (Lipschitz constraint)
        # The change in drift should not be disproportionate to the change in state
        state_delta = np.linalg.norm(proposed_state - current_state)
        drift_delta = abs(drift_proposed - drift_current)

        if state_delta < 1e-6:
            return 1.0

        lipschitz_ratio = drift_delta / state_delta

        # Coherence decays as Lipschitz ratio exceeds 1.0 (fracture)
        # AND as absolute drift increases (ungroundedness)
        coherence = np.exp(-lipschitz_ratio) * (1.0 / (1.0 + drift_proposed))

        return float(coherence)

    def inject_ontology(self, ontology_vector: np.ndarray) -> None:
        """
        Injects a new ontological frame into the field (Plasticity).
        """
        self.add_anchor(ontology_vector, metadata={"type": "ontology_injection"})

    def apply_intervention(self, intervention_vector: np.ndarray) -> bool:
        """
        Tests if an intervention is valid (grounded) within the field.
        """
        drift = self.get_drift_metric(intervention_vector)
        # If drift is too high, the intervention is "ungrounded" (invalid)
        # Threshold can be dynamic based on spatial_coherence_threshold
        return drift < (1.0 - self.spatial_coherence_threshold)
