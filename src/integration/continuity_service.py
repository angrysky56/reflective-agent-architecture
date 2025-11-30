"""
Continuity Service - Causal Signature Calculation

This module implements the Continuity Service, which calculates "Causal Signatures"
for agents based on their historical impact. This satisfies TKUI Axiom 4 (Continuity)
by providing a temporal identity vector without requiring the Precuneus to be stateful.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..persistence.work_history import WorkHistory
from .continuity_field import ContinuityField

logger = logging.getLogger(__name__)


class ContinuityService:
    """
    Service to calculate and retrieve Causal Signatures for agents.

    Upgraded to use Causal Compression's ContinuityField (Identity Manifold)
    to rigorously track semantic drift.
    """

    def __init__(self, work_history: WorkHistory, decay_factor: float = 0.9, embedding_dim: int = 64, continuity_field: Optional[ContinuityField] = None):
        self.work_history = work_history
        self.decay_factor = decay_factor
        self.embedding_dim = embedding_dim

        # Maintain a Continuity Field (Identity Manifold) for each agent
        self.fields: Dict[str, ContinuityField] = {}
        if continuity_field:
            self.fields["system"] = continuity_field

    def get_causal_signature(self, agent_id: str, dim: int = 64) -> torch.Tensor:
        """
        Compute the Causal Signature for a given agent.

        The signature is a vector representing the agent's accumulated impact,
        modulated by its adherence to its Identity Manifold (Continuity Field).

        Args:
            agent_id: The ID of the agent (e.g., "Director", "Explorer").
            dim: The dimension of the returned vector.

        Returns:
            torch.Tensor: Normalized causal signature vector (dim,).
        """
        # Ensure field exists
        if agent_id not in self.fields:
            self.fields[agent_id] = ContinuityField(embedding_dim=dim)
            # Initialize with a seed anchor if empty
            seed = self._generate_agent_seed(agent_id, dim).numpy()
            self.fields[agent_id].add_anchor(seed)

        # 1. Calculate Base Heuristic Vector (Intended Direction)
        base_vector = self._calculate_heuristic_vector(agent_id, dim)

        # 2. Calculate Drift from Identity Manifold
        # We check if the current trajectory (base_vector) aligns with the agent's established identity
        field = self.fields[agent_id]
        try:
            drift = field.get_drift_metric(base_vector.numpy())
        except ValueError:
            drift = 0.0

        # 3. Modulate based on Drift
        # High drift -> Lower confidence/weight in this signature
        # We use a Gaussian kernel to convert drift distance to a similarity score (0 to 1)
        # sigma controls the tolerance for drift
        sigma = 1.0
        continuity_score = np.exp(-(drift**2) / (2 * sigma**2))

        # Modulate the vector magnitude by the continuity score
        # This tells the Precuneus: "If I am drifting, pay less attention to me."
        modulated_vector = base_vector * float(continuity_score)

        return modulated_vector

    def add_anchor(self, agent_id: str, vector: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Consolidate a state into the agent's Identity Manifold.
        Call this when the agent successfully completes a task or reaches a stable state.
        """
        if agent_id not in self.fields:
            # If passing numpy array, get dim from shape
            dim = vector.shape[0]
            self.fields[agent_id] = ContinuityField(embedding_dim=dim)

        # Handle both Tensor and numpy array
        if hasattr(vector, "detach"):
             vec_np = vector.detach().cpu().numpy()
        else:
             vec_np = vector

        self.fields[agent_id].add_anchor(vec_np)
        logger.info(f"Added anchor to Continuity Field for agent {agent_id}")
        if metadata:
            logger.info(f"Anchor metadata: {metadata}")

    def _calculate_heuristic_vector(self, agent_id: str, dim: int) -> torch.Tensor:
        """
        Calculate the heuristic impact vector based on work history.
        (Refactored from original get_causal_signature)
        """
        history_items = self.work_history.search_history(query=agent_id, limit=50)

        if not history_items:
            # Fallback to seed if no history
            return self._generate_agent_seed(agent_id, dim)

        accumulated_vector = torch.zeros(dim)
        agent_seed = self._generate_agent_seed(agent_id, dim)
        current_momentum = 1.0

        for item in history_items:
            impact_score = self._calculate_impact(item)
            accumulated_vector += agent_seed * impact_score * current_momentum
            current_momentum *= self.decay_factor

        norm = torch.norm(accumulated_vector)
        if norm > 0:
            accumulated_vector = accumulated_vector / norm

        return accumulated_vector

    def _generate_agent_seed(self, agent_id: str, dim: int) -> torch.Tensor:
        """Generate a deterministic random vector for an agent ID."""
        seed = abs(hash(agent_id)) % (2**32)
        torch.manual_seed(seed)
        return torch.randn(dim)

    def _calculate_impact(self, item: Dict) -> float:
        """Estimate the impact of a history item."""
        impact = 0.1
        if "causal_impact" in item and item["causal_impact"]:
            try:
                return float(item["causal_impact"])
            except (ValueError, TypeError):
                pass

        result = item.get("result_summary", "")
        if "success" in result.lower():
            impact += 0.5

        energy = item.get("energy", 0.0)
        if energy < 0.0:
            impact += abs(energy)

        return impact
