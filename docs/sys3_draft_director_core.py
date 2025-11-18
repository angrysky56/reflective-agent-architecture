"""
Director Core: Integration of Monitoring, Search, and Escalation

Implements the full Director loop:
1. Monitor entropy from Processor output (System 1)
2. Detect "clash" (high entropy state)
3. Search Manifold with adaptive beta (System 2)
4. If S2 fails, check escalation criteria
5. Escalate to heavy-duty models (System 3)
6. Return new goal for Pointer update
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

# New imports for escalation
from ..escalation import EscalationContext, EscalationManager, EscalationTrigger
from .entropy_monitor import EntropyMonitor
from .search_mvp import SearchResult, energy_aware_knn_search, knn_search

logger = logging.getLogger(__name__)


@dataclass
class DirectorConfig:
    """Configuration for Director."""

    # Entropy monitoring
    entropy_threshold_percentile: float = 0.75
    entropy_history_size: int = 100
    default_entropy_threshold: float = 2.0

    # Search parameters
    search_k: int = 5
    search_metric: str = "cosine"
    exclude_threshold: float = 0.95
    use_energy_aware_search: bool = True

    # Control parameters
    max_search_attempts: int = 3  # Note: This is now for *internal* S2 search
    min_entropy_reduction: float = 0.1

    # Logging
    log_search_episodes: bool = True
    device: str = "cpu"


class DirectorMVP:
    """
    Director MVP: Metacognitive Monitor + Search Engine + Escalation.

    Orchestrates the full System 1-2-3 reasoning cascade.
    """

    def __init__(
        self,
        manifold,
        config: Optional[DirectorConfig] = None,
        escalation_manager: Optional[EscalationManager] = None,  # NEW
        cwd_bridge=None,  # NEW: Needed for full context
    ):
        """
        Initialize Director.

        Args:
            manifold: Manifold (Modern Hopfield Network) to search
            config: Director configuration
            escalation_manager: (NEW) System 3 Escalation Manager
            cwd_bridge: (NEW) Bridge to CWD for context
        """
        self.manifold = manifold
        self.config = config or DirectorConfig()

        # NEW: System 3 components
        self.escalation_manager = escalation_manager
        self.cwd_bridge = cwd_bridge  # For gathering goal/constraint context
        self._escalation_context: Optional[EscalationContext] = None

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

        """
        return self.monitor.check_logits(logits)

    def search(
        self,
        current_state: torch.Tensor,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[SearchResult]:
        """
        Internal System 2 Search: Search Manifold for alternative goal framing.

        """
        memory_patterns = self.manifold.get_patterns()

        if memory_patterns.shape[0] == 0:
            logger.warning("Cannot search: Manifold has no stored patterns")
            return None

        try:
            if self.config.use_energy_aware_search:
                result = energy_aware_knn_search(
                    current_state=current_state,
                    memory_patterns=memory_patterns,
                    energy_evaluator=self.manifold.energy,
                    k=self.config.search_k,
                    metric=self.config.search_metric,
                    exclude_threshold=self.config.exclude_threshold,
                )
            else:
                result = knn_search(
                    current_state=current_state,
                    memory_patterns=memory_patterns,
                    k=self.config.search_k,
                    metric=self.config.search_metric,
                    exclude_threshold=self.config.exclude_threshold,
                )

            if self.config.log_search_episodes:
                self._log_search_episode(current_state, result, context)

            return result

        except Exception as e:
            logger.error(f"Internal search (S2) failed: {e}")
            return None

    # (NEW) Helper to build the escalation context
    def _get_escalation_context(
        self, current_state: torch.Tensor, entropy_value: float, context: Dict[str, Any]
    ) -> EscalationContext:
        """Initializes or updates the escalation context."""

        # Try to get CWD goals if bridge is available
        active_goals = {}
        if self.cwd_bridge:
            try:
                # This would be an async call in a real scenario
                # For this sketch, we assume it's available or we skip
                pass  # active_goals = await self.cwd_bridge.get_active_goals()
            except Exception:
                logger.warning("Could not fetch CWD goals for S3 context.")

        if self._escalation_context is None:
            # First time this clash is seen, create new context
            logger.debug("Initializing new System 3 escalation context.")
            self._escalation_context = EscalationContext(
                original_problem=context.get("problem_description", "Unknown problem"),
                current_goal_state=current_state,
                goal_description=context.get("goal_description", "Unknown goal"),
                manifold_search_attempts=0,
                cwd_search_attempts=0,  # Assuming CWD search is also tracked
                current_entropy=entropy_value,
                entropy_trajectory=[entropy_value],
                active_goals=active_goals,
                constraints=context.get("constraints", []),
            )
        else:
            # Update existing context for this ongoing clash
            self._escalation_context.current_entropy = entropy_value
            self._escalation_context.entropy_trajectory.append(entropy_value)

        return self._escalation_context

    # --- (MODIFIED) MAIN DIRECTOR LOOP ---
    async def check_and_search(
        self,
        current_state: torch.Tensor,
        processor_logits: torch.Tensor,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[torch.Tensor]:
        """
        Main Director loop: monitor (S1) -> S2 search -> S3 escalation.

        This function is now ASYNC to support escalation.
        """
        # Step 1: Monitor entropy (System 1)
        is_clash, entropy_value = self.check_entropy(processor_logits)

        if context is None:
            context = {}
        context["entropy"] = entropy_value
        context["threshold"] = self.monitor.get_threshold()

        # Step 2: Check if intervention needed
        if not is_clash:
            if self._escalation_context is not None:
                logger.debug("Clash resolved. Clearing S3 escalation context.")
                self._escalation_context = None  # Clash resolved, clear context
            return None

        logger.info(
            f"Clash detected! Entropy={entropy_value:.3f} > "
            f"threshold={self.monitor.get_threshold():.3f}"
        )

        # --- Step 3: Internal System 2 Search (Adaptive Beta) ---

        # (NEW) Get or update escalation context
        s3_context = self._get_escalation_context(current_state, entropy_value, context)
        s3_context.manifold_search_attempts += 1  # Track internal search attempt

        original_beta = self.manifold.beta
        internal_search_result: Optional[SearchResult] = None

        try:
            adaptive_beta = self.manifold.compute_adaptive_beta(
                entropy=entropy_value, max_entropy=None
            )

            logger.info(
                f"S2 Search (Attempt {s3_context.manifold_search_attempts}): "
                f"Setting adaptive beta: {adaptive_beta:.3f}"
            )
            self.manifold.set_beta(adaptive_beta)

            context["adaptive_beta"] = adaptive_beta
            context["original_beta"] = original_beta

            internal_search_result = self.search(current_state, context)

        except Exception as e:
            logger.error(f"S2 Search with adaptive beta failed: {e}")
        finally:
            logger.debug(f"Resetting beta to original value: {original_beta:.3f}")
            self.manifold.set_beta(original_beta)

        # Step 4: Evaluate S2 Search Result
        if internal_search_result is not None:
            logger.info("S2 Search SUCCEEDED. Clash resolved internally.")
            self._escalation_context = None  # Success, clear S3 context
            return internal_search_result.best_pattern

        logger.warning(
            f"S2 Search (Attempt {s3_context.manifold_search_attempts}) FAILED to find alternative."
        )

        # --- Step 5: (NEW) System 3 Escalation Check ---

        if not self.escalation_manager:
            logger.debug("S2 failed, but no escalation manager configured.")
            return None  # S2 failed, no S3 to escalate to

        # Check escalation criteria
        should_escalate, trigger = self.escalation_manager.should_escalate(s3_context)

        if not should_escalate:
            logger.debug("S2 failed, but escalation criteria not met yet. Awaiting next step.")
            return None  # Not time to escalate yet

        logger.warning(
            f"--- SYSTEM 3 ESCALATION --- "
            f"Trigger: {trigger.value}. Escalating to heavy-duty models."
        )

        # --- Step 6: Execute System 3 Escalation ---
        try:
            escalation_result = await self.escalation_manager.escalate(s3_context)

            if escalation_result and escalation_result.new_goal_state is not None:
                logger.info(
                    f"S3 Escalation SUCCEEDED. New goal provided by {escalation_result.model_used}."
                )
                self._escalation_context = None  # Escalation successful, clear context

                # The S3 result must be converted to a SearchResult for logging
                # (Or just return the tensor, as the raa_loop expects)
                return escalation_result.new_goal_state
            else:
                logger.error("S3 Escalation FAILED to return a new goal state.")
                # Context remains "stuck" for next attempt
                return None

        except Exception as e:
            logger.error(f"System 3 Escalation FAILED: {e}")
            return None  # Escalation itself failed

    def _log_search_episode(
        self,
        current_state: torch.Tensor,
        result: SearchResult,
        context: Optional[Dict[str, Any]],
    ) -> None:
        """Log search episode for analysis."""
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
        """Get Director statistics."""
        entropy_stats = self.monitor.get_statistics()

        return {
            "entropy": entropy_stats,
            "num_search_episodes": len(self.search_episodes),
            "config": {
                "search_k": self.config.search_k,
                "metric": self.config.search_metric,
                "threshold_percentile": self.config.entropy_threshold_percentile,
            },
        }

    def reset(self) -> None:
        """Reset Director state."""
        self.monitor.reset()
        self.search_episodes.clear()
        self._escalation_context = None  # NEW
