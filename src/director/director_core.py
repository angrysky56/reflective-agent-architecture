"""
Director Core: Integration of Monitoring and Search

Implements the full Director loop:
1. Monitor entropy from Processor output
2. Detect "clash" (high entropy state)
3. Search Manifold for alternative framing
4. Return new goal for Pointer update

This is the Phase 1 (MVP) implementation following SEARCH_MECHANISM_DESIGN.md.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from .entropy_monitor import EntropyMonitor
from .search_mvp import SearchResult, knn_search

logger = logging.getLogger(__name__)


@dataclass
class DirectorConfig:
    """Configuration for Director."""

    # Entropy monitoring
    entropy_threshold_percentile: float = 0.75
    entropy_history_size: int = 100
    default_entropy_threshold: float = 2.0

    # Search parameters
    search_k: int = 5  # Number of neighbors to retrieve
    search_metric: str = "cosine"  # Distance metric
    exclude_threshold: float = 0.95  # Similarity threshold for excluding current basin

    # Control parameters
    max_search_attempts: int = 3  # Maximum search attempts before giving up
    min_entropy_reduction: float = 0.1  # Minimum entropy reduction to accept new goal

    # Logging
    log_search_episodes: bool = True
    device: str = "cpu"


class DirectorMVP:
    """
    Director MVP: Metacognitive Monitor + Search Engine.

    The Director is the core innovation of RAA. It detects confusion via
    entropy monitoring and triggers search in the Manifold for alternative
    conceptual framings.
    """

    def __init__(self, manifold, config: Optional[DirectorConfig] = None):
        """
        Initialize Director.

        Args:
            manifold: Manifold (Modern Hopfield Network) to search
            config: Director configuration
        """
        self.manifold = manifold
        self.config = config or DirectorConfig()

        # Entropy monitor
        self.monitor = EntropyMonitor(
            threshold_percentile=self.config.entropy_threshold_percentile,
            history_size=self.config.entropy_history_size,
            default_threshold=self.config.default_entropy_threshold,
        )

        # Search episode logging
        self.search_episodes = []

    def check_entropy(self, logits: torch.Tensor) -> tuple[bool, float]:
        """
        Check if entropy indicates a clash.

        Args:
            logits: Processor output logits (batch, vocab_size) or (batch, seq_len, vocab_size)

        Returns:
            is_clash: Whether clash was detected
            entropy_value: Computed entropy
        """
        return self.monitor.check_logits(logits)

    def search(
        self,
        current_state: torch.Tensor,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[SearchResult]:
        """
        Search Manifold for alternative goal framing.

        Args:
            current_state: Current goal/state embedding
            context: Optional context information for logging

        Returns:
            SearchResult if alternative found, None if search failed
        """
        # Get patterns from Manifold
        memory_patterns = self.manifold.get_patterns()

        if memory_patterns.shape[0] == 0:
            logger.warning("Cannot search: Manifold has no stored patterns")
            return None

        try:
            # Perform k-NN search
            result = knn_search(
                current_state=current_state,
                memory_patterns=memory_patterns,
                k=self.config.search_k,
                metric=self.config.search_metric,
                exclude_threshold=self.config.exclude_threshold,
            )

            # Log search episode
            if self.config.log_search_episodes:
                self._log_search_episode(current_state, result, context)

            return result

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return None

    def check_and_search(
        self,
        current_state: torch.Tensor,
        processor_logits: torch.Tensor,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[torch.Tensor]:
        """
        Main Director loop: monitor → detect → search → return new goal.

        This is the primary interface used by the RAA integration.

        Args:
            current_state: Current goal state from Pointer
            processor_logits: Output logits from Processor
            context: Optional context (query, history, etc.)

        Returns:
            new_goal: New goal vector if search successful, None if no intervention needed
        """
        # Step 1: Monitor entropy
        is_clash, entropy_value = self.check_entropy(processor_logits)

        # Add entropy to context for logging
        if context is None:
            context = {}
        context['entropy'] = entropy_value
        context['threshold'] = self.monitor.get_threshold()

        # Step 2: Check if intervention needed
        if not is_clash:
            logger.debug(f"No clash detected (entropy={entropy_value:.3f})")
            return None

        logger.info(
            f"Clash detected! Entropy={entropy_value:.3f} > "
            f"threshold={self.monitor.get_threshold():.3f}"
        )

        # Step 3: Search for alternative
        search_result = self.search(current_state, context)

        if search_result is None:
            logger.warning("Search failed to find alternative")
            return None

        # Step 4: Return new goal
        new_goal = search_result.best_pattern

        logger.info(
            f"Search successful. Selected neighbor with score={search_result.selection_score:.3f}"
        )

        return new_goal

    def _log_search_episode(
        self,
        current_state: torch.Tensor,
        result: SearchResult,
        context: Optional[Dict[str, Any]],
    ) -> None:
        """Log search episode for analysis and future policy learning."""
        episode = {
            'current_state': current_state.detach().cpu(),
            'new_state': result.best_pattern.detach().cpu(),
            'neighbor_indices': result.neighbor_indices,
            'distances': result.neighbor_distances.detach().cpu(),
            'selection_score': result.selection_score,
            'entropy': context.get('entropy') if context else None,
            'threshold': context.get('threshold') if context else None,
        }

        self.search_episodes.append(episode)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get Director statistics.

        Returns:
            Dictionary with entropy stats and search episode count
        """
        entropy_stats = self.monitor.get_statistics()

        return {
            'entropy': entropy_stats,
            'num_search_episodes': len(self.search_episodes),
            'config': {
                'search_k': self.config.search_k,
                'metric': self.config.search_metric,
                'threshold_percentile': self.config.entropy_threshold_percentile,
            }
        }

    def reset(self) -> None:
        """Reset Director state."""
        self.monitor.reset()
        self.search_episodes.clear()

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"DirectorMVP(threshold={stats['entropy']['threshold']:.3f}, "
            f"searches={stats['num_search_episodes']})"
        )
