"""
Director Interoception: Computational Interoception for RAA
============================================================

Measures the geometric distance between intent (goal) and result (output)
in embedding space. This provides TRUE interoceptive awareness - knowing
if outputs align with goals at a vector level, not just statistical level.

Key Insight: Entropy monitors OUTPUT uncertainty (exteroception).
Adjunction Tension monitors INTENT-RESULT alignment (interoception).

Formula: Stress = Utility × Distance(Goal, Result)

Author: RAA Enhancement Project
Date: December 10, 2025
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional

import torch
import torch.nn.functional as functional

if TYPE_CHECKING:
    from reflective_agent_architecture.integration.precuneus_integrator import PrecuneusIntegrator
    from reflective_agent_architecture.substrate.goal_controller import GoalController

logger = logging.getLogger(__name__)


@dataclass
class AdjunctionTension:
    """
    Measures the distance between Heuristic (LLM) and Formal (Prover) spaces.

    In category-theoretic terms, this is the "counit" of the adjunction
    between the neural embedding functor and the symbolic grounding functor.
    High tension means the adjunction is "straining" - the two spaces disagree.
    """

    goal_vector: torch.Tensor  # What we intended (from GoalController)
    result_vector: torch.Tensor  # What we produced (embedded result)
    distance: float  # ||goal - result|| (cosine distance, 0-1)
    utility: float  # Value of the result (from evaluator, 0-1)
    stress: float  # utility × distance (high = discrepancy with important result)

    def is_hallucinating(self, entropy: float, threshold: float = 0.3) -> bool:
        """
        Hallucination detection: Low entropy (confident) + High tension (misaligned).

        This is the key diagnostic that entropy alone cannot provide.
        """
        return entropy < 0.3 and self.distance > threshold

    def is_genuinely_complex(self, entropy: float) -> bool:
        """
        Genuine complexity: High entropy (uncertain) + Low tension (aligned).

        The agent is correctly uncertain about a complex domain.
        """
        return entropy > 0.6 and self.distance < 0.3


class DirectorInteroception:
    """
    Direct measurement of internal state geometry.

    Unlike entropy monitoring (which measures output token distributions),
    this class measures the RELATIONSHIP between what the agent intended
    and what it produced - the core of computational self-awareness.

    Design Principle: This is NOT a replacement for entropy monitoring.
    It's a COMPLEMENTARY signal that explains WHY entropy is high/low.
    """

    def __init__(
        self,
        goal_controller: Optional["GoalController"] = None,
        precuneus: Optional["PrecuneusIntegrator"] = None,
        embedding_fn: Optional[Callable[[str], torch.Tensor]] = None,
    ) -> None:
        """
        Initialize interoception module.

        Args:
            goal_controller: GoalController (Pointer) for current goal vector
            precuneus: PrecuneusIntegrator for embedding results
            embedding_fn: Optional override for embedding function
        """
        self.goal_controller = goal_controller
        self.precuneus = precuneus

        # Use provided embedding or fall back to precuneus
        if embedding_fn is not None:
            self.embed = embedding_fn
        elif precuneus is not None and hasattr(precuneus, "embed"):
            self.embed = precuneus.embed
        else:
            self.embed = self._dummy_embed

        self.tension_history: List[float] = []
        self.stress_history: List[float] = []

    def _dummy_embed(self, text: str) -> torch.Tensor:
        """Fallback embedding when no real embedder is available."""
        logger.warning("Using dummy embedding - tension measurements will be unreliable")
        return torch.randn(1, 384)  # Random vector, standard embedding dim

    def measure_tension(
        self, result: str, utility: float = 0.5, goal_override: Optional[torch.Tensor] = None
    ) -> AdjunctionTension:
        """
        Compute distance between goal and result in embedding space.

        Args:
            result: The produced output (text to embed)
            utility: Value/importance of this result (0-1)
            goal_override: Optional explicit goal vector (bypasses goal_controller)

        Returns:
            AdjunctionTension containing distance, utility, and stress metrics
        """
        # 1. Get current goal vector
        if goal_override is not None:
            goal_vec = goal_override
        elif self.goal_controller is not None:
            try:
                goal_vec = self.goal_controller.get_current_goal()
            except Exception as e:
                logger.warning(f"Failed to get goal vector: {e}")
                goal_vec = torch.zeros(1, 384)
        else:
            logger.warning("No goal_controller available - using zero vector")
            goal_vec = torch.zeros(1, 384)

        # 2. Embed the result
        try:
            result_vec = self.embed(result)
        except Exception as e:
            logger.warning(f"Failed to embed result: {e}")
            result_vec = torch.zeros_like(goal_vec)

        # Ensure matching dimensions
        if goal_vec.dim() == 1:
            goal_vec = goal_vec.unsqueeze(0)
        if result_vec.dim() == 1:
            result_vec = result_vec.unsqueeze(0)

        # Match dimensions if they differ
        if goal_vec.shape[-1] != result_vec.shape[-1]:
            logger.warning(f"Dimension mismatch: goal={goal_vec.shape}, result={result_vec.shape}")
            # Pad or truncate to match
            min_dim = min(goal_vec.shape[-1], result_vec.shape[-1])
            goal_vec = goal_vec[..., :min_dim]
            result_vec = result_vec[..., :min_dim]

        # 3. Compute cosine distance
        cos_sim = functional.cosine_similarity(goal_vec, result_vec, dim=-1)
        distance = 1.0 - cos_sim.mean().item()  # 0 = aligned, 1 = orthogonal

        # Clamp to valid range
        distance = max(0.0, min(1.0, distance))

        # 4. Stress = tension when result has high utility but misaligns
        # High utility + high distance = high stress (important result is wrong)
        # Low utility + high distance = low stress (unimportant result is wrong)
        stress = utility * distance

        # Track history
        self.tension_history.append(distance)
        self.stress_history.append(stress)

        return AdjunctionTension(
            goal_vector=goal_vec,
            result_vector=result_vec,
            distance=distance,
            utility=utility,
            stress=stress,
        )

    def get_average_tension(self, window: int = 10) -> float:
        """Get rolling average of tension values."""
        if not self.tension_history:
            return 0.0
        recent = self.tension_history[-window:]
        return sum(recent) / len(recent)

    def get_average_stress(self, window: int = 10) -> float:
        """Get rolling average of stress values."""
        if not self.stress_history:
            return 0.0
        recent = self.stress_history[-window:]
        return sum(recent) / len(recent)

    def get_tension_trend(self, window: int = 10) -> float:
        """
        Get trend of tension (positive = increasing, negative = decreasing).

        Returns difference between recent average and older average.
        """
        if len(self.tension_history) < window * 2:
            return 0.0

        recent = self.tension_history[-window:]
        older = self.tension_history[-window * 2 : -window]

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)

        return recent_avg - older_avg

    def get_statistics(self) -> dict:
        """Get summary statistics for integration with DirectorState."""
        return {
            "tension_avg": self.get_average_tension(),
            "stress_avg": self.get_average_stress(),
            "tension_trend": self.get_tension_trend(),
            "observations": len(self.tension_history),
        }

    def clear_history(self) -> None:
        """Reset history (e.g., after sleep cycle)."""
        self.tension_history.clear()
        self.stress_history.clear()


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == "__main__":
    # Demo without real components
    interoception = DirectorInteroception()

    # Simulate some measurements
    for i in range(5):
        result = f"Test result {i}"
        utility = 0.5 + (i * 0.1)
        tension = interoception.measure_tension(result, utility)
        print(f"Result {i}: Distance={tension.distance:.3f}, Stress={tension.stress:.3f}")

    print(f"\nStatistics: {interoception.get_statistics()}")
