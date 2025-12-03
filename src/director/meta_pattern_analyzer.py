"""
Meta-Pattern Analyzer

Analyzes intervention history to discover patterns in success/failure.
This is the "analytical engine" of the Recursive Observer, detecting
correlations between Layer 4 interventions and Layer 3 outcomes.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .intervention_tracker import InterventionRecord

logger = logging.getLogger(__name__)


@dataclass
class PatternInsight:
    """A discovered pattern that suggests a modification."""

    pattern_type: str  # e.g., "threshold_optimization", "state_specific"
    confidence: float  # 0.0 to 1.0
    recommendation: str  # Human-readable description
    suggested_adjustment: Dict[str, Any]  # Machine-readable parameters
    evidence_count: int  # Number of data points supporting this

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_type": self.pattern_type,
            "confidence": self.confidence,
            "recommendation": self.recommendation,
            "suggested_adjustment": self.suggested_adjustment,
            "evidence_count": self.evidence_count
        }


class MetaPatternAnalyzer:
    """
    Analyzes intervention history to find meta-patterns.

    Key analyses:
    1. Threshold Effectiveness: Does higher/lower entropy threshold improve success?
    2. State Specificity: Do certain cognitive states need different thresholds?
    3. Intervention Type: Which intervention types work best when?
    """

    def __init__(self, min_samples: int = 20):
        """
        Initialize analyzer.

        Args:
            min_samples: Minimum records needed to generate insights
        """
        self.min_samples = min_samples

    def analyze(self, history: List[InterventionRecord]) -> List[PatternInsight]:
        """
        Run all analyses on the provided history.

        Args:
            history: List of intervention records

        Returns:
            List of discovered insights
        """
        # Filter for completed interventions
        completed = [r for r in history if r.outcome is not None]

        if len(completed) < self.min_samples:
            logger.debug(f"Insufficient samples for analysis: {len(completed)} < {self.min_samples}")
            return []

        insights = []

        # 1. Analyze global threshold effectiveness
        threshold_insight = self._analyze_threshold_effectiveness(completed)
        if threshold_insight:
            insights.append(threshold_insight)

        # 2. Analyze state-specific patterns
        state_insights = self._analyze_state_specific_patterns(completed)
        insights.extend(state_insights)

        return insights

    def _analyze_threshold_effectiveness(self, records: List[InterventionRecord]) -> Optional[PatternInsight]:
        """
        Determine if the global entropy threshold should be raised or lowered.

        Logic:
        - If low-threshold interventions often fail (false positives), raise threshold.
        - If high-entropy states often lead to failure when NOT intervened (false negatives - harder to detect here),
          or if successful interventions tend to happen at lower entropies, might lower threshold.

        Simplified Logic for Phase 2:
        - Calculate success rate for interventions triggered at different entropy levels.
        - If interventions at low entropy (e.g. < 1.5) have low 'outcome_quality' (meaning they were likely unnecessary or disruptive), suggest raising threshold.
        - If interventions at high entropy (> 2.5) have high 'outcome_quality', suggests we are catching them correctly, but maybe we should catch them sooner?
        """
        # Extract data: (entropy_threshold_used, outcome_quality)
        data = [(r.entropy_threshold_used, r.outcome.outcome_quality) for r in records]

        if not data:
            return None

        thresholds, _ = zip(*data)

        # Split into "low threshold" and "high threshold" groups relative to median
        median_threshold = np.median(thresholds)
        low_group = [q for t, q in data if t < median_threshold]
        high_group = [q for t, q in data if t >= median_threshold]

        if not low_group or not high_group:
            return None

        avg_low = np.mean(low_group)
        avg_high = np.mean(high_group)

        # Heuristic: If higher thresholds yield better quality, we might be intervening too aggressively
        # (i.e., low threshold interventions are noise/false positives)
        if avg_high > avg_low + 0.1:  # Significant difference
            return PatternInsight(
                pattern_type="threshold_optimization",
                confidence=0.7,  # Moderate confidence
                recommendation=f"Raise threshold. Higher thresholds ({median_threshold:.2f}+) yield better outcomes ({avg_high:.2f} vs {avg_low:.2f}).",
                suggested_adjustment={"threshold_multiplier": 1.1},  # Suggest 10% increase
                evidence_count=len(records)
            )

        # Conversely, if lower thresholds yield better quality, maybe we should be more sensitive
        if avg_low > avg_high + 0.1:
            return PatternInsight(
                pattern_type="threshold_optimization",
                confidence=0.7,
                recommendation=f"Lower threshold. Lower thresholds (<{median_threshold:.2f}) yield better outcomes ({avg_low:.2f} vs {avg_high:.2f}).",
                suggested_adjustment={"threshold_multiplier": 0.9},  # Suggest 10% decrease
                evidence_count=len(records)
            )

        return None

    def _analyze_state_specific_patterns(self, records: List[InterventionRecord]) -> List[PatternInsight]:
        """
        Analyze if specific cognitive states (e.g., 'Looping', 'Focused') have different optimal thresholds.
        """
        insights = []

        # Group by cognitive state
        by_state: Dict[str, List[InterventionRecord]] = {}
        for r in records:
            state = r.cognitive_state_before
            if state not in by_state:
                by_state[state] = []
            by_state[state].append(r)

        for state, state_records in by_state.items():
            if len(state_records) < 5:  # Need minimal samples per state
                continue

            # Calculate average quality for this state
            avg_quality = np.mean([r.outcome.outcome_quality for r in state_records])

            # Compare to global average
            global_avg = np.mean([r.outcome.outcome_quality for r in records])

            # If this state performs poorly, maybe it needs a different sensitivity
            if avg_quality < global_avg - 0.15:
                # Poor performance in this state.
                # Heuristic: If "Looping", we usually need to intervene SOONER (lower threshold).
                if "Looping" in state or "Stuck" in state:
                    insights.append(PatternInsight(
                        pattern_type=f"state_specific_{state}",
                        confidence=0.6,
                        recommendation=f"State '{state}' has poor outcomes ({avg_quality:.2f}). Suggest increasing sensitivity.",
                        suggested_adjustment={
                            "state": state,
                            "threshold_multiplier": 0.8  # Lower threshold significantly
                        },
                        evidence_count=len(state_records)
                    ))

            # If "Flow" or "Focused", maybe we intervene too much?
            elif "Flow" in state or "Focused" in state:
                if avg_quality < global_avg: # Even slightly below average might mean disruption
                     insights.append(PatternInsight(
                        pattern_type=f"state_specific_{state}",
                        confidence=0.6,
                        recommendation=f"State '{state}' might be disrupted by interventions. Suggest decreasing sensitivity.",
                        suggested_adjustment={
                            "state": state,
                            "threshold_multiplier": 1.2  # Raise threshold
                        },
                        evidence_count=len(state_records)
                    ))

        return insights
