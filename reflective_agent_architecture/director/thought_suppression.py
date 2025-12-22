"""
Thought Suppression Module

Implements the "Police Strategy" from Experiment B (Thermodynamic Survival).
Active suppression of high-entropy thoughts that prevents propagation through
the cognitive graph without generating counter-thoughts (Tit-for-Tat trap).

Key Principle: Suppress entropic patterns, don't mirror them.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SuppressionStrategy(Enum):
    """Strategy for handling high-entropy thoughts."""

    PASSIVE = "passive"  # Saint: Tolerate (no action)
    SUPPRESS = "suppress"  # Police: Block propagation (active)
    QUARANTINE = "quarantine"  # Fallback: Isolate when energy-limited


@dataclass
class SuppressionResult:
    """Result of a suppression attempt."""

    strategy: SuppressionStrategy
    suppressed: bool
    energy_cost: float
    entropy_before: float
    entropy_after: Optional[float] = None
    reason: str = ""


class ThoughtSuppressor:
    """
    Active thought suppression mechanism.

    Based on Experiment B findings:
    - Saint (passive tolerance) → System collapse
    - Tit-for-Tat (reactive defection) → Accelerated collapse (Scorched Earth)
    - Police (active suppression) → System stability

    The suppressor:
    1. Identifies high-entropy thought-nodes
    2. Blocks their propagation in the graph
    3. Pays energy cost for suppression
    4. Falls back to quarantine if energy-depleted
    """

    def __init__(
        self,
        suppression_threshold: float = 0.6,
        suppression_cost: float = 2.0,
        quarantine_threshold: float = 0.8,
        quarantine_cost: float = 0.5,
    ):
        """
        Initialize thought suppressor.

        Args:
            suppression_threshold: Entropy level above which to suppress
            suppression_cost: Energy cost to actively suppress
            quarantine_threshold: Entropy level for forced quarantine
            quarantine_cost: Energy cost for quarantine (cheaper than suppression)
        """
        self.suppression_threshold = suppression_threshold
        self.suppression_cost = suppression_cost
        self.quarantine_threshold = quarantine_threshold
        self.quarantine_cost = quarantine_cost

        # Track suppression history
        self.suppression_history: List[SuppressionResult] = []

    def evaluate_thought(
        self,
        thought_id: str,
        entropy: float,
        energy_budget: float,
        graph_handle: Optional[Any] = None,
    ) -> SuppressionResult:
        """
        Evaluate whether a thought should be suppressed.

        Args:
            thought_id: ID of the thought-node in graph
            entropy: Entropy level of the thought
            energy_budget: Available energy for suppression
            graph_handle: Optional handle to graph for actual suppression

        Returns:
            SuppressionResult with action taken
        """

        # Case 1: Low entropy - passive tolerance (Saint mode)
        if entropy < self.suppression_threshold:
            return SuppressionResult(
                strategy=SuppressionStrategy.PASSIVE,
                suppressed=False,
                energy_cost=0.0,
                entropy_before=entropy,
                reason="Entropy below threshold - tolerating",
            )

        # Case 2: Critical entropy - forced quarantine regardless of energy
        if entropy >= self.quarantine_threshold:
            if energy_budget >= self.quarantine_cost:
                if graph_handle:
                    self._quarantine_thought(thought_id, graph_handle)

                return SuppressionResult(
                    strategy=SuppressionStrategy.QUARANTINE,
                    suppressed=True,
                    energy_cost=self.quarantine_cost,
                    entropy_before=entropy,
                    reason="Critical entropy - quarantined",
                )
            else:
                logger.warning(
                    f"Insufficient energy for quarantine of critical thought {thought_id} "
                    f"(need {self.quarantine_cost}, have {energy_budget})"
                )
                return SuppressionResult(
                    strategy=SuppressionStrategy.PASSIVE,
                    suppressed=False,
                    energy_cost=0.0,
                    entropy_before=entropy,
                    reason="Energy depleted - cannot quarantine",
                )

        # Case 3: High entropy - active suppression (Police mode)
        if energy_budget >= self.suppression_cost:
            if graph_handle:
                self._suppress_thought(thought_id, graph_handle)

            return SuppressionResult(
                strategy=SuppressionStrategy.SUPPRESS,
                suppressed=True,
                energy_cost=self.suppression_cost,
                entropy_before=entropy,
                reason="High entropy - actively suppressed",
            )

        # Case 4: High entropy but insufficient energy - try cheaper quarantine
        elif energy_budget >= self.quarantine_cost:
            if graph_handle:
                self._quarantine_thought(thought_id, graph_handle)

            return SuppressionResult(
                strategy=SuppressionStrategy.QUARANTINE,
                suppressed=True,
                energy_cost=self.quarantine_cost,
                entropy_before=entropy,
                reason="High entropy + low energy - quarantined instead",
            )

        # Case 5: Insufficient energy for any action
        else:
            logger.warning(
                f"Insufficient energy to suppress high-entropy thought {thought_id} "
                f"(need {self.suppression_cost}, have {energy_budget})"
            )
            return SuppressionResult(
                strategy=SuppressionStrategy.PASSIVE,
                suppressed=False,
                energy_cost=0.0,
                entropy_before=entropy,
                reason="Energy depleted - cannot suppress",
            )

    def _suppress_thought(self, thought_id: str, graph_handle: Any) -> None:
        """
        Actively suppress a thought by blocking its propagation.

        CRITICAL: This does NOT generate a counter-thought (Tit-for-Tat trap).
        Instead, it marks the node as suppressed, preventing downstream queries
        from retrieving or integrating it.

        Similar to immune system marking a pathogen, not fighting fire with fire.
        """
        try:
            # Mark node as suppressed in graph
            graph_handle.set_node_property(thought_id, "suppressed", True)

            # Optionally: Reduce its retrieval weight
            graph_handle.set_node_property(thought_id, "retrieval_weight", 0.0)

            logger.info(f"Suppressed thought {thought_id} (active policing)")

        except Exception as e:
            logger.error(f"Failed to suppress thought {thought_id}: {e}")

    def _quarantine_thought(self, thought_id: str, graph_handle: Any) -> None:
        """
        Quarantine a thought (cheaper than full suppression).

        Quarantined thoughts are marked but not fully blocked.
        They can still be retrieved with explicit override, but are
        excluded from default queries.
        """
        try:
            # Mark as quarantined
            graph_handle.set_node_property(thought_id, "quarantined", True)

            # Reduce but don't zero retrieval weight
            graph_handle.set_node_property(thought_id, "retrieval_weight", 0.1)

            logger.info(f"Quarantined thought {thought_id} (energy-limited policing)")

        except Exception as e:
            logger.error(f"Failed to quarantine thought {thought_id}: {e}")

    def get_statistics(self) -> Dict:
        """Get suppression statistics."""
        if not self.suppression_history:
            return {"total_evaluations": 0, "suppression_rate": 0.0, "total_energy_spent": 0.0}

        total = len(self.suppression_history)
        suppressed = sum(1 for r in self.suppression_history if r.suppressed)
        total_energy = sum(r.energy_cost for r in self.suppression_history)

        by_strategy = {}
        for strategy in SuppressionStrategy:
            count = sum(1 for r in self.suppression_history if r.strategy == strategy)
            by_strategy[strategy.value] = count

        return {
            "total_evaluations": total,
            "suppression_rate": suppressed / total if total > 0 else 0.0,
            "total_energy_spent": total_energy,
            "by_strategy": by_strategy,
        }
