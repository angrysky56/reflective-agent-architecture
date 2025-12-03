"""
Adaptive Criterion

Manages the dynamic criteria for intervention (e.g., entropy thresholds).
It receives insights from the MetaPatternAnalyzer and safely updates
the system's sensitivity, ensuring stability and preventing runaway feedback.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

from .meta_pattern_analyzer import PatternInsight

logger = logging.getLogger(__name__)


@dataclass
class CriterionState:
    """Persistable state of the adaptive criterion."""
    base_threshold: float
    min_threshold: float
    max_threshold: float
    state_overrides: Dict[str, float] = field(default_factory=dict)
    last_update_timestamp: float = 0.0
    update_count: int = 0


class AdaptiveCriterion:
    """
    Manages the adaptive intervention criteria.

    Responsibilities:
    1. Maintain the current 'base_threshold' for entropy.
    2. Apply updates from PatternInsights (with bounds checking).
    3. Provide the correct threshold for a given cognitive state.
    4. Persist learned criteria to disk.
    """

    def __init__(
        self,
        initial_threshold: float = 2.0,
        min_threshold: float = 0.5,
        max_threshold: float = 4.0,
        persistence_path: Optional[Path] = None
    ):
        self.persistence_path = persistence_path or Path.home() / ".raa" / "adaptive_criterion.json"

        # Initialize state
        self.state = CriterionState(
            base_threshold=initial_threshold,
            min_threshold=min_threshold,
            max_threshold=max_threshold
        )

        # Load if exists
        if self.persistence_path.exists():
            self._load()

    def get_threshold(self, cognitive_state: str = "Neutral") -> float:
        """
        Get the active threshold for the current state.

        Args:
            cognitive_state: Current cognitive state (e.g., 'Looping', 'Flow')

        Returns:
            The entropy threshold to use.
        """
        # Check for state-specific override
        # We check for partial matches (e.g. "Looping" matches "Looping (Stuck)")
        for key, value in self.state.state_overrides.items():
            if key in cognitive_state:
                return value

        return self.state.base_threshold

    def update(self, insight: PatternInsight) -> bool:
        """
        Apply a pattern insight to update the criteria.

        Args:
            insight: The pattern discovered by MetaPatternAnalyzer

        Returns:
            True if an update was applied, False otherwise.
        """
        if insight.confidence < 0.5:
            logger.info(f"Ignoring low confidence insight: {insight.recommendation}")
            return False

        changed = False

        if insight.pattern_type == "threshold_optimization":
            multiplier = insight.suggested_adjustment.get("threshold_multiplier", 1.0)
            new_threshold = self.state.base_threshold * multiplier

            # Clamp to bounds
            new_threshold = max(self.state.min_threshold, min(self.state.max_threshold, new_threshold))

            if abs(new_threshold - self.state.base_threshold) > 0.01:
                logger.info(f"Updating base threshold: {self.state.base_threshold:.2f} -> {new_threshold:.2f}")
                self.state.base_threshold = new_threshold
                changed = True

        elif insight.pattern_type.startswith("state_specific"):
            state_key = insight.suggested_adjustment.get("state")
            multiplier = insight.suggested_adjustment.get("threshold_multiplier", 1.0)

            if state_key:
                # Calculate new value based on current base (or existing override)
                current_val = self.state.state_overrides.get(state_key, self.state.base_threshold)
                new_val = current_val * multiplier

                # Clamp
                new_val = max(self.state.min_threshold, min(self.state.max_threshold, new_val))

                logger.info(f"Updating override for '{state_key}': {current_val:.2f} -> {new_val:.2f}")
                self.state.state_overrides[state_key] = new_val
                changed = True

        if changed:
            self.state.update_count += 1
            self._save()
            return True

        return False

    def _save(self):
        """Save state to disk."""
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persistence_path, 'w') as f:
                json.dump(asdict(self.state), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save adaptive criterion: {e}")

    def _load(self):
        """Load state from disk."""
        try:
            with open(self.persistence_path, 'r') as f:
                data = json.load(f)
                # Safely load known fields
                self.state = CriterionState(
                    base_threshold=data.get("base_threshold", 2.0),
                    min_threshold=data.get("min_threshold", 0.5),
                    max_threshold=data.get("max_threshold", 4.0),
                    state_overrides=data.get("state_overrides", {}),
                    last_update_timestamp=data.get("last_update_timestamp", 0.0),
                    update_count=data.get("update_count", 0)
                )
        except Exception as e:
            logger.error(f"Failed to load adaptive criterion: {e}")
