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
from typing import Any

import torch

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
        # TODO: Implement in Phase 3
        raise NotImplementedError("Phase 3: Utility-biased search")


def utility_biased_energy(
    state: torch.Tensor,
    pattern: torch.Tensor,
    manifold,
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
    # TODO: Implement in Phase 3
    raise NotImplementedError("Phase 3: Utility-biased energy function")
