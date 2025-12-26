import logging
from typing import Dict, Optional, Tuple

import numpy as np

from reflective_agent_architecture.cognition.plasticity_gate import PlasticityGate
from reflective_agent_architecture.integration.continuity_field import ContinuityField

logger = logging.getLogger(__name__)


class StereoscopicEngine:
    """
    The Stereoscopic Engine orchestrates the Dual-Layer Architecture defined in TKUI.
    It integrates the 'Unconditioned Condition' (Code Length/Constraints) and
    'Ground of Being' (Generative Function) through the 'Continuity Field'.

    Components:
    - Continuity Field (Meta Layer): Maintains identity coherence.
    - Plasticity Gate (Base Layer Control): Regulates structural changes.
    - Generative Function: The active agent/mechanism proposing changes.
    """

    def __init__(self, embedding_dim: int = 64, neo4j_uri: Optional[str] = None):
        self.embedding_dim = embedding_dim

        # Initialize Dual-Layer Components
        self.continuity_field = ContinuityField(embedding_dim=embedding_dim, neo4j_uri=neo4j_uri)
        self.plasticity_gate = PlasticityGate()

        # State
        self.current_state = np.zeros(embedding_dim)

    def initialize_state(self, initial_vector: np.ndarray) -> None:
        """Initialize the engine with a starting state."""
        if initial_vector.shape != (self.embedding_dim,):
            raise ValueError(f"Initial vector shape mismatch. Expected ({self.embedding_dim},)")

        self.current_state = initial_vector
        # Anchor the initial state
        self.continuity_field.add_anchor(initial_vector, metadata={"type": "initialization"})
        logger.info("StereoscopicEngine initialized and anchored.")

    def process_intervention(
        self, intervention_vector: np.ndarray, context: str = ""
    ) -> Tuple[bool, float, str]:
        """
        Apply a 'Top-Down Modulation' (Intervention) to the system.
        This acts as a Hypothesis Test against the Continuity Field.

        Args:
            intervention_vector: The proposed change vector (Delta Theta).
            context: Description of the intervention reason.

        Returns:
            (success: bool, score: float, message: str)
        """
        # 1. Evaluate via Plasticity Gate
        # This checks if the intervention is valid given current uncertainty and identity constraints
        permit, gating_score = self.plasticity_gate.evaluate_modification(
            current_state=self.current_state,
            proposed_change=intervention_vector,
            continuity_field=self.continuity_field,
        )

        if not permit:
            msg = f"Intervention rejected by Plasticity Gate. Score: {gating_score:.3f}"
            logger.info(msg)
            return False, gating_score, msg

        # 2. Apply the Intervention (Update State)
        # If permitted, the state evolves
        new_state = self.current_state + intervention_vector

        # 3. Integrate into Continuity Field (Meta-Cognitive Closure)
        # "The act of transformation is integrated into the field"
        self.continuity_field.add_anchor(
            new_state,
            metadata={
                "type": "intervention",
                "context": context,
                "gating_score": gating_score,
                "causal_signature": float(
                    np.linalg.norm(intervention_vector)
                ),  # Simplified signature
            },
        )

        self.current_state = new_state
        msg = f"Intervention accepted. State evolved. Score: {gating_score:.3f}"
        logger.info(msg)

        return True, gating_score, msg

    def get_identity_metrics(self) -> Dict[str, float]:
        """Return current identity/continuity metrics."""
        drift = self.continuity_field.get_drift_metric(self.current_state)
        return {
            "drift": drift,
            "anchors": len(self.continuity_field.anchors),
            "modifications": len(self.plasticity_gate.modification_history),
        }
