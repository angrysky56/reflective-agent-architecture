"""
Adaptive Criterion

Manages the dynamic criteria for intervention (e.g., entropy thresholds, search parameters).
It receives insights from the MetaPatternAnalyzer and safely updates
the system's sensitivity and search strategy.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .meta_pattern_analyzer import PatternInsight

logger = logging.getLogger(__name__)


@dataclass
class ParameterConfig:
    """Configuration for a single adaptive parameter."""
    name: str
    value: Union[float, int, str]
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    param_type: str = "float"  # float, int, categorical
    history: List[Any] = field(default_factory=list)
    state_overrides: Dict[str, Any] = field(default_factory=dict)

    def set_value(self, new_value: Any):
        """Set value with type checking and clamping."""
        if self.param_type == "float":
            val = float(new_value)
            if self.min_value is not None:
                val = max(self.min_value, val)
            if self.max_value is not None:
                val = min(self.max_value, val)
            self.value = val
        elif self.param_type == "int":
            val = int(round(float(new_value)))
            if self.min_value is not None:
                val = max(int(self.min_value), val)
            if self.max_value is not None:
                val = min(int(self.max_value), val)
            self.value = val
        else:
            self.value = new_value

        self.history.append(self.value)
        # Keep history manageable
        if len(self.history) > 100:
            self.history.pop(0)


@dataclass
class CriterionState:
    """Persistable state of the adaptive criterion."""
    parameters: Dict[str, ParameterConfig] = field(default_factory=dict)
    last_update_timestamp: float = 0.0
    update_count: int = 0


class AdaptiveCriterion:
    """
    Manages the adaptive intervention criteria.

    Responsibilities:
    1. Maintain dynamic parameters (thresholds, search k, etc.).
    2. Apply updates from PatternInsights.
    3. Provide correct parameter values for cognitive states.
    4. Persist learned criteria to disk.
    """

    def __init__(
        self,
        persistence_path: Optional[Path] = None
    ):
        self.persistence_path = persistence_path or Path.home() / ".raa" / "adaptive_criterion.json"

        # Initialize default state
        self.state = CriterionState()
        self._initialize_defaults()

        # Load if exists
        if self.persistence_path.exists():
            self._load()

    def _initialize_defaults(self):
        """Initialize default parameters."""
        defaults = [
            ParameterConfig(
                name="entropy_threshold",
                value=2.0,
                min_value=0.5,
                max_value=4.0,
                param_type="float"
            ),
            ParameterConfig(
                name="search_k",
                value=5,
                min_value=1,
                max_value=20,
                param_type="int"
            ),
            ParameterConfig(
                name="search_depth",
                value=3,
                min_value=1,
                max_value=10,
                param_type="int"
            ),
            ParameterConfig(
                name="search_metric",
                value="cosine",
                param_type="categorical"
            )
        ]

        for param in defaults:
            if param.name not in self.state.parameters:
                self.state.parameters[param.name] = param

    def get_parameter(self, name: str, cognitive_state: str = "Neutral") -> Any:
        """
        Get the active value for a parameter, respecting state overrides.
        """
        if name not in self.state.parameters:
            logger.warning(f"Requested unknown parameter: {name}")
            return None

        param = self.state.parameters[name]

        # Check for state-specific override
        for key, value in param.state_overrides.items():
            if key in cognitive_state:
                return value

        return param.value

    def get_threshold(self, cognitive_state: str = "Neutral") -> float:
        """Legacy wrapper for entropy_threshold."""
        return float(self.get_parameter("entropy_threshold", cognitive_state))

    def update(self, insight: PatternInsight) -> bool:
        """
        Apply a pattern insight to update the criteria.
        """
        if insight.confidence < 0.5:
            return False

        changed = False

        # Handle legacy threshold optimization
        if insight.pattern_type == "threshold_optimization":
            multiplier = insight.suggested_adjustment.get("threshold_multiplier", 1.0)
            param = self.state.parameters["entropy_threshold"]
            current_val = param.value
            new_val = current_val * multiplier

            if abs(new_val - current_val) > 0.01:
                param.set_value(new_val)
                logger.info(f"Updated entropy_threshold: {current_val:.2f} -> {param.value:.2f}")
                changed = True

        # Handle generic parameter updates
        elif insight.pattern_type == "parameter_optimization":
            param_name = insight.suggested_adjustment.get("parameter")
            if param_name and param_name in self.state.parameters:
                param = self.state.parameters[param_name]

                # Check for direct value or multiplier
                value = insight.suggested_adjustment.get("value")
                multiplier = insight.suggested_adjustment.get("multiplier")

                new_val = None
                if value is not None:
                    new_val = value
                elif multiplier is not None:
                    new_val = param.value * multiplier
                elif insight.suggested_adjustment.get("is_multiplier", False):
                     # Legacy/Alternative format support
                     adj = insight.suggested_adjustment.get("value") # Here value is the multiplier
                     if adj is not None:
                         new_val = param.value * adj

                if new_val is not None:
                    param.set_value(new_val)
                    logger.info(f"Updated {param_name}: {param.value}")
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
        """Load state from disk with migration support."""
        try:
            with open(self.persistence_path, 'r') as f:
                data = json.load(f)

            # Migration: Check if this is the old format (has 'base_threshold')
            if "base_threshold" in data:
                logger.info("Migrating legacy AdaptiveCriterion format...")
                # Migrate base_threshold
                self.state.parameters["entropy_threshold"].value = data.get("base_threshold", 2.0)
                self.state.parameters["entropy_threshold"].min_value = data.get("min_threshold", 0.5)
                self.state.parameters["entropy_threshold"].max_value = data.get("max_threshold", 4.0)

                # Migrate overrides
                overrides = data.get("state_overrides", {})
                self.state.parameters["entropy_threshold"].state_overrides = overrides

                self.state.update_count = data.get("update_count", 0)
                self.state.last_update_timestamp = data.get("last_update_timestamp", 0.0)

                # Save immediately to update format
                self._save()
            else:
                # Load new format
                # We need to reconstruct ParameterConfig objects
                loaded_params = {}
                for name, p_data in data.get("parameters", {}).items():
                    loaded_params[name] = ParameterConfig(**p_data)

                self.state.parameters = loaded_params
                self.state.update_count = data.get("update_count", 0)
                self.state.last_update_timestamp = data.get("last_update_timestamp", 0.0)

                # Ensure defaults exist if new params were added since save
                self._initialize_defaults()

        except Exception as e:
            logger.error(f"Failed to load adaptive criterion: {e}")
