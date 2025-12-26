"""
Reflexive Closure Engine

The "heart" of the self-modifying architecture.
Orchestrates the loop between Observation (Tracker), Analysis (Analyzer),
and Adaptation (Criterion).
"""

import logging
import random
from typing import Any, Optional

from .adaptive_criterion import AdaptiveCriterion
from .intervention_tracker import InterventionTracker
from .meta_pattern_analyzer import MetaPatternAnalyzer

logger = logging.getLogger(__name__)


class ReflexiveClosureEngine:
    """
    Orchestrates the reflexive closure loop.

    This component is responsible for:
    1. Managing the lifecycle of the sub-components (Tracker, Analyzer, Criterion).
    2. Triggering periodic self-analysis.
    3. Implementing exploration strategies (epsilon-greedy) to gather counterfactuals.
    """

    def __init__(
        self,
        tracker: Optional[InterventionTracker] = None,
        analyzer: Optional[MetaPatternAnalyzer] = None,
        criterion: Optional[AdaptiveCriterion] = None,
        analysis_interval: int = 50,  # Analyze every N interventions
        exploration_rate: float = 0.05,  # 5% chance to perturb threshold
    ):
        self.tracker = tracker or InterventionTracker()
        self.analyzer = analyzer or MetaPatternAnalyzer()
        self.criterion = criterion or AdaptiveCriterion()

        self.analysis_interval = analysis_interval
        self.exploration_rate = exploration_rate

        self._interventions_since_analysis = 0

    def get_parameter(self, name: str, cognitive_state: str = "Neutral") -> Any:
        """
        Get the current value for a parameter, potentially exploring.

        Args:
            name: Parameter name (e.g., 'entropy_threshold', 'search_k')
            cognitive_state: Current cognitive state

        Returns:
            The parameter value (possibly perturbed for exploration)
        """
        base_value = self.criterion.get_parameter(name, cognitive_state)

        if base_value is None:
            return None

        # Epsilon-greedy exploration
        if random.random() < self.exploration_rate:  # nosec B311
            if isinstance(base_value, float):
                # Perturb float by +/- 20%
                perturbation = random.uniform(0.8, 1.2)  # nosec B311
                explored_value = base_value * perturbation
                logger.info(
                    f"Reflexive Exploration ({name}): {base_value:.2f} -> {explored_value:.2f}"
                )
                return explored_value

            elif isinstance(base_value, int):
                # Perturb int by +/- 1 step (or more for larger values)
                step = max(1, int(base_value * 0.2))
                delta = random.choice([-step, step])  # nosec B311
                explored_value = max(1, base_value + delta)  # Assuming positive ints usually
                logger.info(f"Reflexive Exploration ({name}): {base_value} -> {explored_value}")
                return explored_value

            # Categorical/String exploration not yet implemented

        return base_value

    def get_threshold(self, cognitive_state: str = "Neutral") -> float:
        """
        Legacy wrapper for entropy threshold.
        """
        val = self.get_parameter("entropy_threshold", cognitive_state)
        return float(val) if val is not None else 2.0

    def record_intervention_start(self, **kwargs: Any) -> str:
        """Proxy to tracker.start_intervention."""
        return self.tracker.start_intervention(**kwargs).episode_id

    def record_intervention_end(self, **kwargs: Any) -> None:
        """
        Proxy to tracker.finish_intervention, but also triggers analysis loop.
        """
        self.tracker.finish_intervention(**kwargs)
        self._interventions_since_analysis += 1

        if self._interventions_since_analysis >= self.analysis_interval:
            self._run_reflexive_analysis()

    def _run_reflexive_analysis(self) -> None:
        """
        Run the meta-analysis and update criteria.
        This is the "Reflexive Closure" moment where the system modifies itself.
        """
        logger.info("Running Reflexive Closure Analysis...")

        # 1. Get recent history
        # We look at more than just the interval to have robust stats
        history = self.tracker.get_recent_interventions(limit=self.analysis_interval * 2)

        # 2. Analyze
        insights = self.analyzer.analyze(history)

        if not insights:
            logger.info("No actionable patterns found.")
            self._interventions_since_analysis = 0
            return

        # 3. Adapt
        updates_applied = 0
        for insight in insights:
            if self.criterion.update(insight):
                updates_applied += 1
                logger.info(f"Reflexive Update Applied: {insight.recommendation}")

        if updates_applied > 0:
            logger.info(f"Reflexive Closure Complete: Applied {updates_applied} updates.")
        else:
            logger.info(
                "Reflexive Closure Complete: No updates applied (low confidence or bounds)."
            )

        self._interventions_since_analysis = 0
