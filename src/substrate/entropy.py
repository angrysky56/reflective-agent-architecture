import math
from enum import Enum
from typing import Any, Dict, List


class CognitiveState(Enum):
    FOCUS = "focus"  # Low entropy, optimization, convergence
    EXPLORE = "explore"  # High entropy, adaptation, divergence
    SLEEP = "sleep"  # Maintenance, consolidation


class EntropyMonitor:
    """
    Monitors system entropy to drive cognitive state transitions.

    Entropy is calculated based on the diversity of active concepts or tools.
    High Entropy -> Chaos/Confusion -> Needs FOCUS (Convergence)
    Low Entropy -> Stagnation/Boredom -> Needs EXPLORE (Divergence)
    """

    def __init__(self, focus_threshold: float = 2.5, explore_threshold: float = 1.0):
        self.focus_threshold = focus_threshold  # Above this -> Trigger FOCUS
        self.explore_threshold = explore_threshold  # Below this -> Trigger EXPLORE
        self.current_entropy = 0.0
        self.history: List[float] = []
        self.state = CognitiveState.EXPLORE  # Default start state

    def calculate_entropy(self, distribution: List[float]) -> float:
        """
        Calculate Shannon entropy of a probability distribution.
        H(X) = -sum(p(x) * log2(p(x)))
        """
        if not distribution:
            return 0.0

        entropy = 0.0
        for p in distribution:
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    def update_from_counts(self, counts: Dict[str, int]) -> float:
        """
        Update entropy based on a frequency count of active concepts/tools.
        """
        total = sum(counts.values())
        if total == 0:
            return 0.0

        distribution = [count / total for count in counts.values()]
        self.current_entropy = self.calculate_entropy(distribution)
        self.history.append(self.current_entropy)

        # Determine state
        self._update_state()

        return self.current_entropy

    def _update_state(self) -> None:
        """Update cognitive state based on entropy thresholds."""
        # Hysteresis could be added here to prevent rapid switching
        if self.current_entropy > self.focus_threshold:
            self.state = CognitiveState.FOCUS
        elif self.current_entropy < self.explore_threshold:
            self.state = CognitiveState.EXPLORE
        # Else maintain current state (hysteresis/stability)

    def get_baseline_entropy(self, window: int = 10) -> float:
        """Calculate moving average of entropy over the last `window` steps."""
        if not self.history:
            return 0.0
        relevant_history = self.history[-window:]
        return sum(relevant_history) / len(relevant_history)

    def get_entropy_trend(self, window: int = 5) -> str:
        """
        Analyze the trend of entropy over the last `window` steps.
        Returns: "increasing", "decreasing", or "stable"
        """
        if len(self.history) < 2:
            return "stable"

        relevant_history = self.history[-window:]
        if len(relevant_history) < 2:
            return "stable"

        # Simple linear regression slope or just start-end comparison
        # Using start-end for simplicity and robustness
        start = relevant_history[0]
        end = relevant_history[-1]
        diff = end - start

        if diff > 0.1:
            return "increasing"
        elif diff < -0.1:
            return "decreasing"
        else:
            return "stable"

    def get_status(self) -> Dict[str, Any]:
        return {
            "entropy": self.current_entropy,
            "state": self.state.value,
            "baseline": self.get_baseline_entropy(),
            "trend": self.get_entropy_trend(),
            "thresholds": {"focus": self.focus_threshold, "explore": self.explore_threshold},
        }
