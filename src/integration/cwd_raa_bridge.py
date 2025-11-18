"""
CWD-RAA Bridge: Main integration orchestrator

This module is the central coordinator for RAA-CWD integration.
It manages:
1. Tool library synchronization with Manifold
2. Entropy monitoring of CWD operations
3. Triggering RAA search on confusion
4. Routing alternatives back to CWD
5. Recording integration metrics

Usage:
    bridge = CWDRAABridge(cwd_server, raa_director, manifold)

    # Execute monitored CWD operation
    result = bridge.execute_monitored_operation(
        operation='hypothesize',
        params={'node_a_id': 'n1', 'node_b_id': 'n2'}
    )
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from .embedding_mapper import EmbeddingMapper
from .entropy_calculator import EntropyCalculator, cwd_to_logits

logger = logging.getLogger(__name__)


@dataclass
class BridgeConfig:
    """Configuration for CWD-RAA bridge."""

    # Embedding settings
    embedding_dim: int = 512
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Entropy monitoring
    entropy_threshold: float = 2.0  # Will be adaptive
    enable_monitoring: bool = True

    # Search triggering
    max_search_attempts: int = 3
    search_on_confusion: bool = True

    # Metrics
    log_integration_events: bool = True
    device: str = "cpu"


class CWDRAABridge:
    """
    Main integration coordinator between CWD and RAA.

    Phase 1: Infrastructure + basic monitoring
    Phase 2: Entropy-triggered search
    Phase 3: Utility-biased search
    Phase 4: Bidirectional learning
    """

    def __init__(
        self,
        cwd_server,  # CWD server instance (or API client)
        raa_director,  # RAA Director instance
        manifold,  # RAA Manifold (Hopfield network)
        config: Optional[BridgeConfig] = None,
        pointer: Optional[Any] = None,  # Optional GoalController (Pointer)
        get_current_goal: Optional[Callable[[], torch.Tensor]] = None,  # Optional goal provider
    ):
        """
        Initialize integration bridge.

        Args:
            cwd_server: CWD server/client for operations
            raa_director: RAA Director for monitoring + search
            manifold: RAA Manifold for pattern storage
            config: Bridge configuration
        """
        self.cwd_server = cwd_server
        self.raa_director = raa_director
        self.manifold = manifold
        self.config = config or BridgeConfig()
        self.pointer = pointer
        self._get_current_goal_fn = get_current_goal

        # Initialize components
        self.embedding_mapper = EmbeddingMapper(
            embedding_dim=self.config.embedding_dim,
            model_name=self.config.embedding_model,
            device=self.config.device,
        )

        self.entropy_calculator = EntropyCalculator()

        # Tool-pattern mapping (to be implemented)
        self.tool_to_pattern_idx: dict[str, int] = {}
        self.pattern_idx_to_tool: dict[int, str] = {}

        # Integration metrics
        self.metrics = {
            "operations_monitored": 0,
            "entropy_spikes_detected": 0,
            "searches_triggered": 0,
            "alternatives_found": 0,
            "integration_events": [],
        }

        logger.info("CWDRAABridge initialized successfully")

    def _get_current_goal(self) -> Optional[torch.Tensor]:
        """Retrieve current goal embedding from integration context.

        Priority:
        1) Explicit callback `get_current_goal`
        2) Pointer reference via `GoalController.get_current_goal()`
        Returns first item if batched.
        """
        try:
            goal: Optional[torch.Tensor] = None
            if self._get_current_goal_fn is not None:
                goal = self._get_current_goal_fn()
            elif self.pointer is not None and hasattr(self.pointer, "get_current_goal"):
                goal = self.pointer.get_current_goal()

            if goal is None:
                return None

            # Ensure 1D embedding for search
            if goal.dim() == 2:
                goal = goal[0]
            return goal
        except Exception as e:
            logger.warning(f"Unable to retrieve current goal: {e}")
            return None

    def execute_monitored_operation(
        self,
        operation: str,
        params: dict[str, Any],
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Execute CWD operation with RAA monitoring.

        Phase 1: Just execute and monitor (no intervention)
        Phase 2: Trigger RAA search on high entropy

        Args:
            operation: CWD operation name ('hypothesize', 'synthesize', etc.)
            params: Parameters for the operation

        Returns:
            Operation result (potentially modified by RAA)
        """
        self.metrics["operations_monitored"] += 1

        # Execute CWD operation
        logger.debug(f"Executing CWD operation: {operation}")
        result = self._execute_cwd_operation(operation, params)

        if not self.config.enable_monitoring:
            return result

        # Convert result to logits for entropy calculation (unifies CWD + RAA)
        logits = cwd_to_logits(operation, result, self.entropy_calculator)

        # Delegate clash detection to Director's adaptive monitor
        is_clash, entropy = self.raa_director.check_entropy(logits)

        logger.debug(f"Operation entropy: {entropy:.3f} (clash={is_clash})")

        # Check if entropy indicates confusion
        if is_clash:
            self.metrics["entropy_spikes_detected"] += 1
            logger.info(f"High entropy detected: {entropy:.3f}")

            if self.config.search_on_confusion:
                # Phase 2: Trigger RAA search (placeholder for now)
                self.metrics["searches_triggered"] += 1

                current_goal = self._get_current_goal()
                if current_goal is None:
                    logger.info(
                        "Search skipped: current goal unavailable (provide pointer or callback)"
                    )
                else:
                    try:
                        search_result = self.raa_director.search(current_goal)
                        if search_result is not None:
                            # Update goal if Pointer available
                            if self.pointer is not None and hasattr(self.pointer, "set_goal"):
                                self.pointer.set_goal(search_result.best_pattern)
                                self.metrics["alternatives_found"] += 1
                                logger.info(
                                    "Pointer goal updated from search result (Phase 2 stub)"
                                )
                            else:
                                logger.info(
                                    "Search result available, but no Pointer to update goal"
                                )
                    except Exception as e:
                        logger.warning(f"Search attempt failed: {e}")

        # Log integration event
        if self.config.log_integration_events:
            self._log_event(operation, result, entropy)

        return result

    def sync_tools_to_manifold(self) -> int:
        """
        Synchronize CWD tool library with RAA Manifold.

        For each compressed tool in CWD:
        1. Convert tool to embedding vector
        2. Store as pattern in Manifold
        3. Record bidirectional mapping

        Returns:
            Number of tools synchronized
        """
        # TODO: Implement after CWD integration is available
        logger.warning("sync_tools_to_manifold not yet implemented")
        return 0

    def _execute_cwd_operation(
        self,
        operation: str,
        params: dict[str, Any],
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Execute CWD operation (abstracted for testing).

        In production, this calls actual CWD server/API if available.
        Fallback returns mock data for local testing.
        """
        # Prefer direct call into provided CWD server/client
        try:
            if self.cwd_server is not None and hasattr(self.cwd_server, operation):
                op_fn = getattr(self.cwd_server, operation)
                if callable(op_fn):
                    return op_fn(**params)  # type: ignore[misc]
        except Exception as e:
            logger.warning(f"CWD operation '{operation}' failed, falling back to mock: {e}")

        logger.warning(f"Mock CWD execution for {operation}")

        if operation == "hypothesize":
            return [
                {"hypothesis": "Mock hypothesis 1", "confidence": 0.8},
                {"hypothesis": "Mock hypothesis 2", "confidence": 0.4},
            ]
        elif operation == "synthesize":
            return {
                "synthesis": "Mock synthesis",
                "quality": {"coverage": 0.9, "coherence": 0.7, "fidelity": 0.8},
            }
        elif operation == "constrain":
            return {
                "valid": True,
                "satisfaction_score": 0.9,
            }
        else:
            return {}

    def _log_event(
        self,
        operation: str,
        result: Any,
        entropy: float,
    ) -> None:
        """Log integration event for analysis."""
        event = {
            "operation": operation,
            "entropy": entropy,
            "triggered_search": entropy > self.config.entropy_threshold,
        }
        self.metrics["integration_events"].append(event)

    def get_metrics(self) -> dict[str, Any]:
        """Get integration metrics."""
        return self.metrics.copy()

    def reset_metrics(self) -> None:
        """Reset metrics counters."""
        self.metrics = {
            "operations_monitored": 0,
            "entropy_spikes_detected": 0,
            "searches_triggered": 0,
            "alternatives_found": 0,
            "integration_events": [],
        }
