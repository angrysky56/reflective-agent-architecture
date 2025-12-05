"""
Plasticity Modulator

Controls the integration bandwidth (P parameter) of the Precuneus.

Based on refined phenomenal time equation:
    t_phen = E x P

Where:
    E = Prediction Error (entropy)
    P = Manifold Plasticity (integration rate)

This resolves paradoxes:
- Anesthesia: E > 0, P -> 0 => t_phen = 0 (no experience)
- Flow state: E low, P high => t_phen significant (rich experience)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PlasticityState:
    """Current plasticity state."""
    value: float  # 0.0 to 1.0
    tau_integration: float  # Integration timescale in seconds
    mode: str  # 'sluggish', 'normal', 'flow'
    confidence: float


class PlasticityModulator:
    """
    Modulates integration bandwidth based on cognitive state.

    P = f(energy, confidence, manifold_state)

    High P (fast integration):
    - Flow states (high confidence + low entropy)
    - Exploration mode
    - Learning opportunities

    Low P (slow integration):
    - Uncertainty (low confidence)
    - Energy depletion
    - Stuck states (high entropy)
    """

    def __init__(
        self,
        P_min: float = 0.1,
        P_max: float = 1.0,
        tau_min: float = 0.1,  # Fast integration (100ms)
        tau_max: float = 2.0,  # Slow integration (2s)
    ):
        self.P_min = P_min
        self.P_max = P_max
        self.tau_min = tau_min
        self.tau_max = tau_max

        # Current state
        self.current_P = 0.5
        self.current_tau = 1.0

    def compute_P(
        self,
        energy: float,
        confidence: float,
        entropy: float,
        manifold_stability: float = 1.0
    ) -> PlasticityState:
        """
        Compute current plasticity value.

        Args:
            energy: Current energy budget (0-100)
            confidence: Current confidence (0-1)
            entropy: Current entropy (0-inf, typically 0-5)
            manifold_stability: Stability metric from manifold (0-1)

        Returns:
            PlasticityState with computed P value
        """

        # Normalize inputs
        energy_norm = np.clip(energy / 100.0, 0, 1)
        entropy_norm = np.clip(entropy / 5.0, 0, 1)

        # Compute P using weighted factors
        # High P when: high energy, high confidence, low entropy, high stability
        P = (
            0.3 * energy_norm +
            0.3 * confidence +
            0.2 * (1 - entropy_norm) +
            0.2 * manifold_stability
        )

        # Clip to bounds
        P = np.clip(P, self.P_min, self.P_max)

        # Compute integration timescale
        # P = 1/tau, so tau = 1/P (with bounds)
        tau = np.interp(P, [self.P_min, self.P_max], [self.tau_max, self.tau_min])

        # Determine mode
        if P < 0.3:
            mode = 'sluggish'
        elif P > 0.7:
            mode = 'flow'
        else:
            mode = 'normal'

        # Update state
        self.current_P = P
        self.current_tau = tau

        # Compute confidence in this P value
        confidence_in_P = np.prod([energy_norm, confidence, manifold_stability])

        return PlasticityState(
            value=P,
            tau_integration=tau,
            mode=mode,
            confidence=confidence_in_P
        )

    def apply_to_precuneus(self, precuneus, P_state: PlasticityState):
        """
        Apply plasticity modulation to Precuneus gating.

        The Precuneus uses P to modulate its integration gate:
        - High P -> More signals integrated (wider gate)
        - Low P -> Fewer signals integrated (narrow gate)
        """
        # This would modify the Precuneus energy_threshold or integration rate
        # Actual implementation depends on Precuneus API

        if hasattr(precuneus, 'set_integration_rate'):
            precuneus.set_integration_rate(P_state.value)
            logger.info(f"Precuneus plasticity set to {P_state.value:.2f} ({P_state.mode})")
        else:
            logger.warning("Precuneus does not support plasticity modulation")
