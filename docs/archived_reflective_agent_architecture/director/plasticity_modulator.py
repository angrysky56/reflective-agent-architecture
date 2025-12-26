"""
plasticity Modulator

Controls the integration bandwidth (p parameter) of the precuneus.

Based on refined phenomenal time equation:
    t_phen = E x p

Where:
    E = prediction Error (entropy)
    p = Manifold plasticity (integration rate)

This resolves paradoxes:
- Anesthesia: E > 0, p -> 0 => t_phen = 0 (no experience)
- Flow state: E low, p high => t_phen significant (rich experience)
"""

import logging
from dataclasses import dataclass
from typing import Any

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

    p = f(energy, confidence, manifold_state)

    High p (fast integration):
    - Flow states (high confidence + low entropy)
    - Exploration mode
    - Learning opportunities

    Low p (slow integration):
    - Uncertainty (low confidence)
    - Energy depletion
    - Stuck states (high entropy)
    """

    def __init__(
        self,
        min_p: float = 0.1,
        max_p: float = 1.0,
        tau_min: float = 0.1,  # Fast integration (100ms)
        tau_max: float = 2.0,  # Slow integration (2s)
    ):
        self.min_p = min_p
        self.max_p = max_p
        self.tau_min = tau_min
        self.tau_max = tau_max

        # Current state
        self.current_p = 0.5
        self.current_tau = 1.0

    def compute_p(
        self, energy: float, confidence: float, entropy: float, manifold_stability: float = 1.0
    ) -> PlasticityState:
        """
        Compute current plasticity value.

        Args:
            energy: Current energy budget (0-100)
            confidence: Current confidence (0-1)
            entropy: Current entropy (0-inf, typically 0-5)
            manifold_stability: Stability metric from manifold (0-1)

        Returns:
            plasticityState with computed p value
        """

        # Normalize inputs
        energy_norm = np.clip(energy / 100.0, 0, 1)
        entropy_norm = np.clip(entropy / 5.0, 0, 1)

        # Compute p using weighted factors
        # High p when: high energy, high confidence, low entropy, high stability
        p = (
            0.3 * energy_norm
            + 0.3 * confidence
            + 0.2 * (1 - entropy_norm)
            + 0.2 * manifold_stability
        )

        # Clip to bounds
        p = np.clip(p, self.min_p, self.max_p)

        # Compute integration timescale
        # p = 1/tau, so tau = 1/p (with bounds)
        tau = np.interp(p, [self.min_p, self.max_p], [self.tau_max, self.tau_min])

        # Determine mode
        if p < 0.3:
            mode = "sluggish"
        elif p > 0.7:
            mode = "flow"
        else:
            mode = "normal"

        # Update state
        self.current_p = p
        self.current_tau = tau

        # Compute confidence in this p value
        confidence_in_p = np.prod([energy_norm, confidence, manifold_stability])

        return PlasticityState(value=p, tau_integration=tau, mode=mode, confidence=confidence_in_p)

    def apply_to_precuneus(self, precuneus: Any, p_state: PlasticityState) -> None:
        """
        Apply plasticity modulation to precuneus gating.

        The precuneus uses p to modulate its integration gate:
        - High p -> More signals integrated (wider gate)
        - Low p -> Fewer signals integrated (narrow gate)
        """
        # This would modify the precuneus energy_threshold or integration rate
        # Actual implementation depends on precuneus ApI

        if hasattr(precuneus, "set_integration_rate"):
            precuneus.set_integration_rate(p_state.value)
            logger.info(f"precuneus plasticity set to {p_state.value:.2f} ({p_state.mode})")
        else:
            logger.warning("precuneus does not support plasticity modulation")
