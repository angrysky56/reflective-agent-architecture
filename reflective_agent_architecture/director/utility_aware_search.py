"""
Utility-Aware Search Module
===========================

Implements the modified Hopfield Energy landscape where "Utility" (Valence + Goal Proximity)
deforms the basin of attraction.

Theoretical Basis:
E'(x) = E_Hopfield(x) - lambda * U(x)

Where U(x) is derived from Director's emotional valence.
Positive Valence -> Deepens Attractors (Stability/Flow)
Negative Valence -> Flattens Landscape (Instability/Change)
"""

from typing import Union

import torch


class UtilityAwareSearch:
    """
    Biases associative memory retrieval using emotional valence.
    """

    def __init__(self, lambda_val: float = 0.5, temperature: float = 1.0):
        self.lambda_val = lambda_val  # Intentionality strength
        self.temperature = temperature

    def compute_biased_energy(
        self, base_energy: Union[float, torch.Tensor], valence: float, goal_alignment: float = 0.0
    ) -> Union[float, torch.Tensor]:
        """
        Compute the modified energy scalar based on valence.

        Args:
            base_energy: Standard Hopfield energy (-Similarity). Lower is better.
            valence: Emotional state [-1.0, 1.0].
            goal_alignment: Proximity to current goal [0.0, 1.0].

        Returns:
            Biased energy value.
        """
        # Utility U(x) composite
        # We weigh Valence heavily as it represents "Internal Truth/Health"
        # We weigh Goal Alignment as "External Progress"

        # If valence is negative (Distress), U approaches negative.
        # If valence is positive (Flow), U approaches positive.
        utility = (0.7 * valence) + (0.3 * goal_alignment)

        # E' = E - lambda * U
        # Higher Utility subtracts from Energy -> Lower Energy State -> Stable Attractor
        biased_energy = base_energy - (self.lambda_val * utility)

        return biased_energy

    def get_landscape_deformation_factor(self, valence: float) -> float:
        """
        Debug helper: How much is the landscape being bent?
        Returns the delta applied to energy.
        """
        # Assuming goal_alignment = 0 for pure affect check
        utility = 0.7 * valence
        return -(self.lambda_val * utility)
