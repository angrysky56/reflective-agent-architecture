"""
Epistemic Discriminator

Implements complexity and randomness estimation from Experiment C.
Enables the Director to distinguish:
- Solvable but complex (-> focused attention)
- Unsolvable due to chaos (-> dissonance trigger)
- Solvable but noisy (-> suppression then solve)
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .epistemic_metrics import estimate_complexity, estimate_randomness

logger = logging.getLogger(__name__)


@dataclass
class EpistemicAssessment:
    """Result of epistemic discrimination."""
    complexity_score: float
    randomness_score: float
    complexity_type: str  # 'simple', 'complex', 'discontinuous'
    randomness_type: str  # 'structured', 'random'
    recommendation: str   # 'attempt_solve', 'focused_search', 'suppress_then_solve', 'trigger_dissonance'
    confidence: float     # How confident in this assessment


class EpistemicDiscriminator:
    """
    Epistemic discrimination engine.

    Estimates:
    1. Complexity: How much structure exists? (learnable but resource-intensive)
    2. Randomness: How much is unpredictable? (unlearnable noise)

    Based on these, recommends cognitive strategy.
    """

    def __init__(
        self,
        complexity_threshold: float = 0.6,
        randomness_threshold: float = 0.2,
        dissonance_threshold: float = 0.5,
    ):
        self.complexity_threshold = complexity_threshold
        self.randomness_threshold = randomness_threshold
        self.dissonance_threshold = dissonance_threshold

    def assess(self, signal: np.ndarray) -> EpistemicAssessment:
        """
        Assess epistemic properties of a signal.

        Args:
            signal: 1D array representing cognitive signal (e.g., entropy history, prediction errors)

        Returns:
            EpistemicAssessment with recommended strategy
        """

        # 1. Estimate complexity
        # We use the promoted epistemic_metrics functions
        # Note: metrics expect list or array
        complexity_info = estimate_complexity(signal)
        complexity_score = complexity_info['complexity_score']
        complexity_type = complexity_info['type']

        # 2. Estimate randomness
        randomness_info = estimate_randomness(signal)
        randomness_score = randomness_info['randomness_score']
        randomness_type = randomness_info['type']

        # 3. Determine recommendation
        recommendation, confidence = self._determine_strategy(
            complexity_score, randomness_score, complexity_type
        )

        return EpistemicAssessment(
            complexity_score=complexity_score,
            randomness_score=randomness_score,
            complexity_type=complexity_type,
            randomness_type=randomness_type,
            recommendation=recommendation,
            confidence=confidence
        )

    def _determine_strategy(
        self,
        complexity: float,
        randomness: float,
        complexity_type: str
    ) -> Tuple[str, float]:
        """
        Determine recommended cognitive strategy.

        Decision logic from Experiment C:
        - High randomness -> Suppress noise first
        - High complexity + low randomness -> Focused attention
        - High both -> Suppress then focused
        - Low both -> Standard solve
        - Discontinuous -> Flag and approximate
        """

        confidence = 1.0 - abs(complexity - 0.5) * abs(randomness - 0.5)

        if complexity_type == 'discontinuous':
            return 'approximate_with_warning', 0.9

        if randomness > self.randomness_threshold:
            if complexity > self.complexity_threshold:
                return 'suppress_then_focused', confidence
            else:
                return 'suppress_then_solve', confidence

        if complexity > self.complexity_threshold:
            return 'focused_search', confidence

        return 'attempt_solve', confidence
