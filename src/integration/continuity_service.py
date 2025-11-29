"""
Continuity Service - Causal Signature Calculation

This module implements the Continuity Service, which calculates "Causal Signatures"
for agents based on their historical impact. This satisfies TKUI Axiom 4 (Continuity)
by providing a temporal identity vector without requiring the Precuneus to be stateful.
"""

import logging
from typing import Dict

import torch

from ..persistence.work_history import WorkHistory

logger = logging.getLogger(__name__)


class ContinuityService:
    """
    Service to calculate and retrieve Causal Signatures for agents.
    """

    def __init__(self, work_history: WorkHistory, decay_factor: float = 0.9):
        self.work_history = work_history
        self.decay_factor = decay_factor  # For temporal momentum

    def get_causal_signature(self, agent_id: str, dim: int = 64) -> torch.Tensor:
        """
        Compute the Causal Signature for a given agent.

        The signature is a vector representing the agent's accumulated impact
        on the system (graph changes, energy reduction, etc.).

        Args:
            agent_id: The ID of the agent (e.g., "Director", "Explorer").
            dim: The dimension of the returned vector.

        Returns:
            torch.Tensor: Normalized causal signature vector (dim,).
        """
        # 1. Retrieve relevant history
        # We look for operations performed by this agent or related to it.
        # Since WorkHistory is generic, we assume 'params' might contain 'agent_id'
        # or we filter by operation type if that maps to agents.
        # For this MVP, we'll search for the agent_id in the params string.

        history_items = self.work_history.search_history(query=agent_id, limit=50)

        if not history_items:
            return torch.zeros(dim)

        # 2. Aggregate Impact
        # We'll construct a pseudo-vector based on the "impact" of each action.
        # Impact = Energy Reduction + Graph Changes (heuristic)

        accumulated_vector = torch.zeros(dim)

        # Create a deterministic seed vector for this agent to represent its "Identity"
        # This ensures that "Explorer" always starts with a distinct base direction.
        agent_seed = self._generate_agent_seed(agent_id, dim)

        current_momentum = 1.0

        for item in history_items:
            # Parse impact from item
            impact_score = self._calculate_impact(item)

            # Add to accumulator: Identity * Impact * Recency
            accumulated_vector += agent_seed * impact_score * current_momentum

            # Decay momentum for older items
            current_momentum *= self.decay_factor

        # 3. Normalize
        norm = torch.norm(accumulated_vector)
        if norm > 0:
            accumulated_vector = accumulated_vector / norm

        return accumulated_vector

    def _generate_agent_seed(self, agent_id: str, dim: int) -> torch.Tensor:
        """Generate a deterministic random vector for an agent ID."""
        # Use hash of agent_id to seed random generator
        seed = abs(hash(agent_id)) % (2**32)
        torch.manual_seed(seed)
        return torch.randn(dim)

    def _calculate_impact(self, item: Dict) -> float:
        """
        Estimate the impact of a history item.
        """
        impact = 0.1  # Base impact for doing anything

        # 1. Check for explicit causal_impact field (to be added to WorkHistory)
        if "causal_impact" in item and item["causal_impact"]:
            try:
                return float(item["causal_impact"])
            except (ValueError, TypeError):
                pass

        # 2. Heuristics based on result summary or diagnostics
        result = item.get("result_summary", "")
        if "success" in result.lower():
            impact += 0.5

        # 3. Energy based
        energy = item.get("energy", 0.0)
        # Lower energy (more stability) implies higher positive impact in some contexts,
        # but here we might view "high energy reduction" as impact.
        # Let's assume the stored 'energy' is the state energy.
        # If it's low (stable), the action was likely good.
        if energy < 0.0: # Negative energy often implies stability in Hopfield nets
            impact += abs(energy)

        return impact
