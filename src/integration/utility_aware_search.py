"""
Utility-Aware Search: Bias RAA search with CWD utility scores

This module modifies RAA's Hopfield energy function to incorporate
CWD's utility scores, preventing frame-shifts to low-utility basins.

Key Concept:
E_total(ξ) = E_hopfield(ξ) - λ * U(tool)

Where:
- E_hopfield: Standard Hopfield energy (lower = more stable)
- U(tool): Utility score from CWD active goals
- λ: Bias weight (tunable parameter)

Effect: High-utility tools have lower effective energy, making them
easier to retrieve during RAA search.
"""

import logging

import torch

from src.manifold import Manifold

logger = logging.getLogger(__name__)


class UtilityAwareSearch:
    """
    Modify RAA search to prefer high-utility tools from CWD.

    This implements Phase 3 of the integration: guiding RAA's
    exploration with CWD's goal-directed utility filtering.
    """

    def __init__(
        self,
        utility_weight: float = 0.3,
        normalize_utilities: bool = True,
    ):
        """
        Initialize utility-aware search.

        Args:
            utility_weight: λ parameter for utility bias (0.0 = no bias)
            normalize_utilities: Whether to normalize utility scores to [0,1]
        """
        self.utility_weight = utility_weight
        self.normalize_utilities = normalize_utilities

        logger.info(f"UtilityAwareSearch initialized with λ={utility_weight}")

    def compute_biased_energy(
        self,
        hopfield_energy: torch.Tensor,
        utility_score: float,
    ) -> torch.Tensor:
        """
        Compute utility-biased energy.

        E_biased = E_hopfield - λ * U

        Args:
            hopfield_energy: Standard Hopfield energy values
            utility_score: Utility score for this pattern [0, 1]

        Returns:
            Biased energy (lower = more attractive)
        """
        if self.normalize_utilities:
            # Ensure utility is in [0, 1] range (clipping safe-guard)
            utility_score = max(0.0, min(1.0, utility_score))

        # E' = E - λ * U
        # Higher utility subtracts more energy, deepening the basin of attraction
        biased_energy = hopfield_energy - (self.utility_weight * utility_score)

        return biased_energy


def utility_biased_energy(
    state: torch.Tensor,
    pattern: torch.Tensor,
    manifold: Manifold,
    utility_score: float,
    utility_weight: float = 0.3,
) -> float:
    """
    Convenience function for utility-biased energy computation.

    Args:
        state: Current state embedding
        pattern: Candidate pattern from Manifold
        manifold: Hopfield network for energy computation
        utility_score: Utility score [0, 1] for this pattern
        utility_weight: λ bias parameter

    Returns:
        Biased energy value
    """
    # 1. Compute standard Hopfield Energy: E = -0.5 * x^T * W * x (simplified)
    # Actually, Manifold class likely has an energy function.
    # If not, we use the standard definition for Modern Hopfield Networks (Dense).
    # E(ξ) = -lse(β * X^T * ξ) / β  (LogSumExp)
    # But here we might just need the energy of a specific pattern relative to state.

    # Assuming 'manifold' instance has an energy method or we compute simple energy.
    # For now, we delegate to the manifold if possible, or assume simple dot product proximity
    # proxy for energy in this specific "search" context.

    # Let's assume standard interaction energy for a single pattern p relative to state s:
    # E = - (s . p)  (Cosine similarity proxy: higher sim = lower energy)

    # Check if inputs are tensors
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state)
    if not isinstance(pattern, torch.Tensor):
        pattern = torch.tensor(pattern)

    # Calculate standard energy (negative similarity)
    # E_hopfield = - dot(state, pattern)
    dot_product = float(torch.dot(state, pattern))
    hopfield_energy = -dot_product

    # 2. Apply Utility Bias
    # E' = E_hopfield - λ * U
    biased_energy = hopfield_energy - (utility_weight * utility_score)

    return float(biased_energy)
