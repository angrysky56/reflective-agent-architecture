"""
Entropy Monitor

Implements Shannon entropy calculation for detecting model confusion.

Based on:
- ERGO: Entropy-guided Resetting for Generation Optimization (ACL 2025)
- Semantic Energy: Detecting LLM Hallucination Beyond Entropy (2025)
"""

import math
from typing import Any, List

import torch
import torch.nn.functional as f


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute Shannon entropy from transformer logits.

    H(p) = -∑ p(x) log p(x)

    Args:
        logits: Raw output from transformer (batch, vocab_size) or (batch, seq_len, vocab_size)

    Returns:
        Entropy value(s). Shape depends on input:
        - (batch,) if input is (batch, vocab_size)
        - (batch, seq_len) if input is (batch, seq_len, vocab_size)
    """
    # Convert logits to probabilities
    probs = f.softmax(logits, dim=-1)
    log_probs = f.log_softmax(logits, dim=-1)

    # Shannon entropy: -sum(p * log(p))
    entropy = -(probs * log_probs).sum(dim=-1)

    return entropy


class EntropyMonitor:
    """
    Monitors entropy to detect "clashes" (high uncertainty states).

    The monitor maintains a history of entropy values and uses adaptive
    thresholding to detect when the model is confused.
    """

    def __init__(
        self,
        threshold_percentile: float = 0.75,
        history_size: int = 100,
        min_samples_for_adaptive: int = 10,
        default_threshold: float = 2.0,
    ):
        """
        Initialize entropy monitor.

        Args:
            threshold_percentile: Percentile of historical entropy to use as threshold
            history_size: Number of recent entropy values to keep
            min_samples_for_adaptive: Minimum samples before using adaptive threshold
            default_threshold: fixed threshold to use before enough samples
        """
        self.threshold_percentile = threshold_percentile
        self.history_size = history_size
        self.min_samples_for_adaptive = min_samples_for_adaptive
        self.default_threshold = default_threshold

        self.entropy_history: List[float] = []

    def add_entropy(self, entropy: float) -> None:
        """Add entropy value to history."""
        self.entropy_history.append(entropy)

        # Keep only recent history
        if len(self.entropy_history) > self.history_size:
            self.entropy_history = self.entropy_history[-self.history_size :]

    def get_threshold(self) -> float:
        """
        Get current adaptive threshold.

        Uses percentile-based threshold once enough samples are collected,
        otherwise uses default threshold.

        Returns:
            Current threshold value
        """
        if len(self.entropy_history) < self.min_samples_for_adaptive:
            return self.default_threshold

        # Compute adaptive threshold from recent history
        threshold = torch.quantile(
            torch.tensor(self.entropy_history), self.threshold_percentile
        ).item()

        return float(threshold)

    def is_clash(self, entropy: float, add_to_history: bool = True) -> bool:
        """
        Detect if current entropy indicates a "clash" (confusion state).

        Args:
            entropy: Current entropy value
            add_to_history: Whether to add this entropy to history

        Returns:
            True if entropy exceeds threshold (clash detected)
        """
        if add_to_history:
            self.add_entropy(entropy)

        threshold = self.get_threshold()
        return entropy > threshold

    def check_logits(self, logits: torch.Tensor) -> tuple[bool, float]:
        """
        Convenience method: compute entropy and check for clash.

        Args:
            logits: Transformer output logits (batch, vocab_size) or (batch, seq_len, vocab_size)

        Returns:
            is_clash: Whether clash was detected
            entropy_value: Computed entropy value
        """
        # Compute entropy
        entropy = compute_entropy(logits)

        # Take mean if multiple values (batch or sequence)
        if entropy.dim() > 0:
            entropy_value = entropy.mean().item()
        else:
            entropy_value = entropy.item()

        # Check for clash
        if math.isnan(entropy_value) or math.isinf(entropy_value):
            entropy_value = 0.0

        is_clash_detected = self.is_clash(entropy_value)

        return is_clash_detected, entropy_value

    def get_statistics(self) -> dict[str, Any]:
        """
        Get statistics about entropy history.

        Returns:
            Dictionary with mean, std, min, max, current threshold
        """
        if not self.entropy_history:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "threshold": self.default_threshold,
                "num_samples": 0,
            }

        history_tensor = torch.tensor(self.entropy_history)

        return {
            "mean": history_tensor.mean().item(),
            "std": history_tensor.std().item(),
            "min": history_tensor.min().item(),
            "max": history_tensor.max().item(),
            "threshold": self.get_threshold(),
            "num_samples": len(self.entropy_history),
        }

    def reset(self) -> None:
        """Clear entropy history."""
        self.entropy_history.clear()

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"EntropyMonitor(threshold={stats['threshold']:.3f}, "
            f"samples={stats['num_samples']}, "
            f"mean={stats['mean']:.3f}±{stats['std']:.3f})"
        )
