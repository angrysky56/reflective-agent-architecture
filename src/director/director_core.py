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
from .search_mvp import SearchResult, energy_aware_knn_search, knn_search
from .sheaf_diagnostics import SheafAnalyzer, SheafConfig, SheafDiagnostics

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
    use_energy_aware_search: bool = True  # Use energy-aware selection (aligns with Hopfield theory)

    # Control parameters
    max_search_attempts: int = 3  # Maximum search attempts before giving up
    min_entropy_reduction: float = 0.1  # Minimum entropy reduction to accept new goal

    # Logging
    log_search_episodes: bool = True
    log_search_episodes: bool = True
    device: str = "cpu"

    # Sheaf Diagnostics
    sheaf_config: Optional[SheafConfig] = None


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

        # Sheaf Analyzer
        sheaf_config = self.config.sheaf_config or SheafConfig(device=self.config.device)
        self.sheaf_analyzer = SheafAnalyzer(sheaf_config)

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
            # Choose search strategy based on configuration
            if self.config.use_energy_aware_search:
                # Energy-aware search: aligns with Hopfield energy landscape
                result = energy_aware_knn_search(
                    current_state=current_state,
                    memory_patterns=memory_patterns,
                    energy_evaluator=self.manifold.energy,
                    k=self.config.search_k,
                    metric=self.config.search_metric,
                    exclude_threshold=self.config.exclude_threshold,
                )
                logger.debug("Used energy-aware search (Hopfield-aligned)")
            else:
                # Basic k-NN search (Phase 1 MVP)
                result = knn_search(
                    current_state=current_state,
                    memory_patterns=memory_patterns,
                    k=self.config.search_k,
                    metric=self.config.search_metric,
                    exclude_threshold=self.config.exclude_threshold,
                )
                logger.debug("Used basic k-NN search")

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
        Main Director loop: monitor → detect → compute adaptive beta → search → return.

        This is the primary interface used by the RAA integration.

        The Director now dynamically adjusts exploration based on confusion:
        - High entropy (confusion) → low beta → more exploratory search
        - Low entropy (confidence) → high beta → more focused search

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
        context["entropy"] = entropy_value
        context["threshold"] = self.monitor.get_threshold()

        # Step 2: Check if intervention needed
        if not is_clash:
            logger.debug(f"No clash detected (entropy={entropy_value:.3f})")
            return None

        logger.info(
            f"Clash detected! Entropy={entropy_value:.3f} > "
            f"threshold={self.monitor.get_threshold():.3f}"
        )

        # --- ADAPTIVE BETA LOGIC ---
        # Store original beta to reset it after search
        original_beta = self.manifold.beta
        search_result = None

        try:
            # Step 3: Compute adaptive beta based on confusion (entropy)
            # Let compute_adaptive_beta calculate max_entropy internally
            adaptive_beta = self.manifold.compute_adaptive_beta(
                entropy=entropy_value, max_entropy=None
            )

            logger.info(
                f"Setting adaptive beta: {adaptive_beta:.3f} (original: {original_beta:.3f}, "
                f"entropy: {entropy_value:.3f})"
            )

            # Temporarily set the manifold's beta for this search
            self.manifold.set_beta(adaptive_beta)

            # Add adaptive beta to context for logging
            context["adaptive_beta"] = adaptive_beta
            context["original_beta"] = original_beta

            # Step 4: Search for alternative using the adaptive beta
            search_result = self.search(current_state, context)

        except Exception as e:
            logger.error(f"Search with adaptive beta failed: {e}")
            # The 'finally' block will still run to clean up beta
        finally:
            # Step 5: CRITICAL - Reset beta to its original value
            # This ensures the next generation step doesn't use the temporary exploratory beta
            logger.debug(f"Resetting beta to original value: {original_beta:.3f}")
            self.manifold.set_beta(original_beta)

        # --- END ADAPTIVE BETA LOGIC ---

        if search_result is None:
            logger.warning("Search failed to find alternative")
            return None

        # Step 6: Return new goal
        new_goal = search_result.best_pattern

        logger.info(
            f"Search successful. Selected neighbor with score={search_result.selection_score:.3f}"
        )

        return new_goal

    def diagnose(
        self,
        weights: list[torch.Tensor],
        target_error: Optional[torch.Tensor] = None,
        feedback_weights: Optional[list[torch.Tensor]] = None,
    ) -> SheafDiagnostics:
        """
        Perform sheaf-theoretic diagnosis of the network.

        Args:
            weights: Network weights
            target_error: Optional target error
            feedback_weights: Optional feedback weights

        Returns:
            SheafDiagnostics result
        """
        return self.sheaf_analyzer.full_diagnosis(weights, target_error, feedback_weights)

    def _log_search_episode(
        self,
        current_state: torch.Tensor,
        result: SearchResult,
        context: Optional[Dict[str, Any]],
    ) -> None:
        """Log search episode for analysis and future policy learning."""
        episode = {
            "current_state": current_state.detach().cpu(),
            "new_state": result.best_pattern.detach().cpu(),
            "neighbor_indices": result.neighbor_indices,
            "distances": result.neighbor_distances.detach().cpu(),
            "selection_score": result.selection_score,
            "entropy": context.get("entropy") if context else None,
            "threshold": context.get("threshold") if context else None,
            "adaptive_beta": context.get("adaptive_beta") if context else None,
            "original_beta": context.get("original_beta") if context else None,
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
            "entropy": entropy_stats,
            "num_search_episodes": len(self.search_episodes),
            "config": {
                "search_k": self.config.search_k,
                "metric": self.config.search_metric,
                "threshold_percentile": self.config.entropy_threshold_percentile,
            },
            "sheaf_diagnostics": {
                "h1_threshold": self.sheaf_analyzer.config.h1_escalation_threshold,
                "overlap_threshold": self.sheaf_analyzer.config.overlap_warning_threshold,
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
