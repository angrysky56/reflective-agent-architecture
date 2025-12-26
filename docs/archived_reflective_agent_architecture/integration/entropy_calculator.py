"""
Entropy Calculator: Convert CWD operations to entropy signals

This module converts CWD reasoning operations into probability distributions
that RAA's Director can monitor for confusion/stuck states.

Key Concept:
CWD operations (hypothesize, synthesize, constrain) produce results with
varying confidence. We convert these confidences into pseudo-probability
distributions, then compute Shannon entropy as a confusion signal.

High entropy = Low confidence/high confusion → Trigger RAA search
Low entropy = High confidence → Continue normal operation
"""

import logging
from typing import Any

import torch
import torch.nn.functional as f

logger = logging.getLogger(__name__)


class EntropyCalculator:
    """
    Converts CWD operation results to entropy signals for RAA monitoring.

    Strategy:
    1. Extract confidence/similarity scores from CWD results
    2. Convert to pseudo-probability distribution
    3. Compute Shannon entropy
    4. Normalize to expected range
    """

    def __init__(
        self,
        temperature: float = 1.0,
        min_entropy: float = 0.0,
        max_entropy: float | None = None,
    ):
        """
        Initialize entropy calculator.

        Args:
            temperature: Softmax temperature for probability conversion
            min_entropy: Minimum expected entropy value
            max_entropy: Maximum expected entropy (auto-computed if None)
        """
        self.temperature = temperature
        self.min_entropy = min_entropy
        self.max_entropy = max_entropy

        logger.info(f"EntropyCalculator initialized with temperature={temperature}")

    def hypothesize_to_logits(
        self,
        hypothesis_results: list[dict[str, Any]],
    ) -> torch.Tensor:
        """
        Convert hypothesize() results to logits for entropy calculation.

        CWD's hypothesize generates multiple hypothesis paths with
        confidence/evidence scores. We convert these to a distribution.

        Args:
            hypothesis_results: List of hypotheses with scores
                [
                    {"hypothesis": "...", "confidence": 0.8, "evidence": [...]},
                    {"hypothesis": "...", "confidence": 0.3, "evidence": [...]},
                    ...
                ]

        Returns:
            Logits tensor of shape (num_hypotheses,)
        """
        if not hypothesis_results:
            logger.warning("Empty hypothesis results, returning uniform logits")
            return torch.zeros(2)  # Minimal distribution

        # Extract confidence scores
        confidences = torch.tensor(
            [h.get("confidence", 0.5) for h in hypothesis_results],
            dtype=torch.float32,
        )

        # Convert to logits (inverse softmax with temperature)
        logits = torch.log(confidences + 1e-8) / self.temperature

        return logits

    def synthesize_to_logits(
        self,
        synthesis_result: dict[str, Any],
    ) -> torch.Tensor:
        """
        Convert synthesize() result to logits.

        Synthesis combines multiple thoughts into one. We measure:
        1. How well the synthesis covers input thoughts (coverage)
        2. How coherent the synthesis is (coherence)
        3. How much information is preserved (fidelity)

        Args:
            synthesis_result: Synthesis output with quality metrics
                {
                    "synthesis": "...",
                    "quality": {
                        "coverage": 0.9,
                        "coherence": 0.7,
                        "fidelity": 0.8
                    }
                }

        Returns:
            Logits representing synthesis quality distribution
        """
        quality = synthesis_result.get("quality", {})

        # Extract metrics (default to medium confidence)
        coverage = quality.get("coverage", 0.5)
        coherence = quality.get("coherence", 0.5)
        fidelity = quality.get("fidelity", 0.5)

        # Create distribution: [good_synthesis, poor_synthesis]
        good_score = (coverage + coherence + fidelity) / 3.0
        poor_score = 1.0 - good_score

        scores = torch.tensor([good_score, poor_score], dtype=torch.float32)
        if torch.isnan(scores).any():
            scores = torch.tensor([0.5, 0.5], dtype=torch.float32)
        logits = torch.log(scores + 1e-8) / self.temperature

        return logits

    def constrain_to_logits(
        self,
        constraint_result: dict[str, Any],
    ) -> torch.Tensor:
        """
        Convert constrain() result to logits.

        Constraint validation checks if a thought satisfies rules.
        We measure how well constraints are satisfied.

        Args:
            constraint_result: Constraint validation output
                {
                    "valid": True/False,
                    "violations": [...],
                    "satisfaction_score": 0.9
                }

        Returns:
            Logits representing constraint satisfaction
        """
        valid = constraint_result.get("valid", True)
        satisfaction = constraint_result.get("satisfaction_score", 1.0 if valid else 0.0)

        # Distribution: [satisfies, violates]
        sat_score = satisfaction
        unsat_score = 1.0 - satisfaction

        scores = torch.tensor([sat_score, unsat_score], dtype=torch.float32)
        if torch.isnan(scores).any():
            scores = torch.tensor([0.5, 0.5], dtype=torch.float32)
        logits = torch.log(scores + 1e-8) / self.temperature

        return logits

    def deconstruct_to_logits(
        self,
        deconstruct_result: dict[str, Any],
    ) -> torch.Tensor:
        """
        Convert deconstruct() result to logits.

        Deconstruction splits a problem. We assume high confidence if successful.
        Future: Measure ambiguity of the split.

        Args:
            deconstruct_result: Deconstruction output

        Returns:
            Logits representing deconstruction confidence
        """
        # Default to high confidence (low entropy) for now
        # Distribution: [confident, uncertain]
        confidence = 0.9
        uncertainty = 0.1

        scores = torch.tensor([confidence, uncertainty], dtype=torch.float32)
        logits = torch.log(scores + 1e-8) / self.temperature

        return logits

    def compute_entropy(self, logits: torch.Tensor) -> float:
        """
        Compute Shannon entropy from logits.

        H(p) = -∑ p(x) log₂ p(x)

        Args:
            logits: Unnormalized log-probabilities

        Returns:
            Entropy value in bits
        """
        # Convert to probabilities
        probs = f.softmax(logits, dim=-1)

        # Compute Shannon entropy
        log_probs = f.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum()

        # Convert to bits (instead of nats)
        entropy_bits = entropy / torch.log(torch.tensor(2.0))

        return float(entropy_bits.item())


def cwd_to_logits(
    operation: str,
    result: dict[str, Any] | list[dict[str, Any]],
    calculator: EntropyCalculator | None = None,
) -> torch.Tensor:
    """
    Convenience function to convert any CWD operation result to logits.

    Args:
        operation: CWD operation name ('hypothesize', 'synthesize', 'constrain')
        result: Operation result
        calculator: Optional custom calculator (creates default if None)

    Returns:
        Logits tensor suitable for RAA entropy monitoring

    Example:
        >>> result = cwd.hypothesize(node_a, node_b)
        >>> logits = cwd_to_logits('hypothesize', result)
        >>> entropy = calculator.compute_entropy(logits)
    """
    if calculator is None:
        calculator = EntropyCalculator()

    if operation == "hypothesize":
        hypothesis_list = result if isinstance(result, list) else [result]
        return calculator.hypothesize_to_logits(hypothesis_list)
    elif operation == "synthesize":
        synthesis_dict = result[0] if isinstance(result, list) else result
        return calculator.synthesize_to_logits(synthesis_dict)
    elif operation == "constrain":
        constraint_dict = result[0] if isinstance(result, list) else result
        return calculator.constrain_to_logits(constraint_dict)
    elif operation == "deconstruct":
        deconstruct_dict = result[0] if isinstance(result, list) else result
        return calculator.deconstruct_to_logits(deconstruct_dict)

    else:
        logger.error(f"Unknown CWD operation: {operation}")
        # Return neutral distribution
        return torch.zeros(2)
