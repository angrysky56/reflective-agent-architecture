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
from typing import Any, Callable, Optional, TypedDict, cast

import torch

from src.persistence.work_history import WorkHistory

from .embedding_mapper import EmbeddingMapper
from .entropy_calculator import EntropyCalculator, cwd_to_logits

logger = logging.getLogger(__name__)


class IntegrationMetrics(TypedDict):
    """Type definition for integration metrics."""

    operations_monitored: int
    entropy_spikes_detected: int
    searches_triggered: int
    alternatives_found: int
    integration_events: list[dict[str, Any]]


@dataclass
class BridgeConfig:
    """Configuration for CWD-RAA bridge.

    Entropy Threshold Guide:
    - Shannon entropy for binary distributions: 0.0-1.0 bits
    - 0.0 bits = perfect certainty [1.0, 0.0]
    - 1.0 bits = maximum confusion [0.5, 0.5]
    - Recommended thresholds:
      * 0.5 bits: Moderate confusion (70/30 split)
      * 0.7 bits: High confusion (60/40 split)
      * 0.9 bits: Very high confusion (near 50/50)
    """

    # Embedding settings
    embedding_dim: int = 512
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Entropy monitoring (FIXED: 0.6 bits is appropriate for binary distributions)
    entropy_threshold: float = 0.6  # Detects moderate-to-high confusion
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
        cwd_server: Any,  # CWD server instance (or API client)
        raa_director: Any,  # RAA Director instance
        manifold: Any,  # RAA Manifold (Hopfield network)
        config: Optional[BridgeConfig] = None,
        pointer: Optional[Any] = None,  # Optional GoalController (Pointer)
        processor: Optional[Any] = None,  # Optional Processor for shadow monitoring
        sleep_cycle: Optional[Any] = None,  # Optional SleepCycle for auto-nap
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
        self.processor = processor
        self.sleep_cycle = sleep_cycle
        self._get_current_goal_fn = get_current_goal

        # Initialize components: reuse CWD server's embedding model if available
        preloaded_model = None
        inferred_dim = self.config.embedding_dim
        try:
            if hasattr(self.cwd_server, "embedding_model"):
                preloaded_model = getattr(self.cwd_server, "embedding_model")
                # Try to derive dimension from model or workspace probe
                if hasattr(preloaded_model, "get_sentence_embedding_dimension"):
                    inferred_dim = int(preloaded_model.get_sentence_embedding_dimension())
                elif hasattr(self.cwd_server, "_embed_text"):
                    sample = self.cwd_server._embed_text("bridge dim probe")
                    inferred_dim = int(len(sample))
        except Exception as e:
            # Fallback to provided config
            logger.debug(
                f"Could not infer embedding dimension from CWD server: {e}. "
                f"Using config default: {inferred_dim}"
            )

        # Update config embedding_dim if inferred from server
        self.config.embedding_dim = inferred_dim

        self.embedding_mapper = EmbeddingMapper(
            embedding_dim=inferred_dim,
            model_name=self.config.embedding_model,
            device=self.config.device,
            preloaded_model=preloaded_model,
        )

        self.entropy_calculator = EntropyCalculator()

        # Tool-pattern mapping (to be implemented)
        self.tool_to_pattern_idx: dict[str, int] = {}
        self.pattern_idx_to_tool: dict[int, str] = {}

        # Integration metrics
        self.metrics: IntegrationMetrics = {
            "operations_monitored": 0,
            "entropy_spikes_detected": 0,
            "searches_triggered": 0,
            "alternatives_found": 0,
            "integration_events": [],
        }

        # Persistence
        self.history = WorkHistory()

        logger.info("CWDRAABridge initialized successfully")

    def set_director(self, director: Any) -> None:
        """Set RAA Director instance (post-initialization)."""
        self.raa_director = director
        logger.info("RAA Director injected into CWDRAABridge")

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

        # Phase 1.5: Shadow Monitoring (Cognitive Proprioception)
        # Run a pass through the Processor to generate attention patterns for the Director
        if self.processor is not None:
            self._run_shadow_monitoring(operation)

        # Log to persistent history
        cognitive_state, energy = self.raa_director.latest_cognitive_state
        diagnostics = getattr(self.raa_director, "latest_diagnostics", {})

        # Phase 7: Energy Recharge Protocol (Auto-Nap)
        # If energy is critically low (e.g., < -0.6), trigger a sleep cycle
        if self.sleep_cycle is not None and energy < -0.6:
            logger.info(f"CRITICAL ENERGY DEPLETION ({energy:.3f}). Triggering Auto-Nap.")
            try:
                nap_results = self.sleep_cycle.dream(epochs=1)
                logger.info(f"Auto-Nap complete: {nap_results['message']}")
                # Re-check state after nap
                cognitive_state, energy = self.raa_director.latest_cognitive_state
            except Exception as e:
                logger.error(f"Auto-Nap failed: {e}")

        self.history.log_operation(
            operation=operation,
            params=params,
            result=result,
            cognitive_state=cognitive_state,
            energy=energy,
            diagnostics=diagnostics,
        )

        return result

    def _run_shadow_monitoring(self, operation: str) -> None:
        """
        Run a shadow pass to populate cognitive state.

        Since the Processor is currently untrained, we simulate meaningful attention
        patterns based on the operation type to demonstrate the Director's capabilities.
        """
        try:
            heads = 8
            seq_len = 16
            device = self.config.device

            # Generate synthetic attention based on operation
            if operation in ["deconstruct", "synthesize", "constrain"]:
                # Focused (Diagonal)
                attention = (
                    torch.eye(seq_len, device=device)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(1, heads, 1, 1)
                )
                # Add noise
                attention = attention + 0.1 * torch.rand_like(attention)
                attention = attention / attention.sum(dim=-1, keepdim=True)

            elif operation in ["hypothesize", "explore_for_utility", "recall_work"]:
                # Broad (Uniform)
                attention = torch.ones((1, heads, seq_len, seq_len), device=device)
                attention = attention / attention.sum(dim=-1, keepdim=True)

            elif operation == "diagnose_pointer":
                # Looping (Off-diagonal) - simulate checking for loops
                attention = torch.zeros((1, heads, seq_len, seq_len), device=device)
                for i in range(seq_len):
                    if i > 0:
                        attention[0, :, i, i - 1] = 1.0
                attention = attention + 0.1 * torch.rand_like(attention)
                attention = attention / (attention.sum(dim=-1, keepdim=True) + 1e-6)

            else:
                # Random/Unknown
                attention = torch.rand((1, heads, seq_len, seq_len), device=device)

            # Monitor the simulated thought
            with torch.no_grad():
                logger.debug(f"Simulating attention for {operation}...")
                self.raa_director.monitor_thought_process(attention)
                logger.debug("Shadow monitoring pass completed.")

        except Exception as e:
            logger.error(f"Shadow monitoring failed: {e}", exc_info=True)

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
        if (
            not self.cwd_server
            or not hasattr(self.cwd_server, "workspace")
            or not self.cwd_server.workspace
        ):
            logger.warning("Cannot sync tools: No workspace available in CWD server")
            return 0

        tool_library = getattr(self.cwd_server.workspace, "tool_library", {})
        if not tool_library:
            return 0

        count = 0
        for tool_id, tool_data in tool_library.items():
            # Skip if already synced (optimization)
            if tool_id in self.tool_to_pattern_idx:
                continue

            embedding = tool_data.get("embedding")
            if not embedding:
                continue

            try:
                # Convert list to tensor
                pattern = torch.tensor(embedding, device=self.config.device)

                # Store in Manifold
                metadata = {
                    "id": tool_id,
                    "name": tool_data.get("name", "unknown"),
                    "type": "tool",
                    "source": "sync",
                }

                # Check if store_pattern accepts metadata (it should)
                if hasattr(self.manifold, "store_pattern"):
                    self.manifold.store_pattern(pattern, metadata=metadata)

                    # Update local mapping
                    # Assuming the new pattern is at the end
                    idx = self.manifold.num_patterns - 1
                    self.tool_to_pattern_idx[tool_id] = idx
                    self.pattern_idx_to_tool[idx] = tool_id

                    count += 1
            except Exception as e:
                logger.warning(f"Failed to sync tool {tool_id}: {e}")

        if count > 0:
            logger.info(f"Synchronized {count} tools to Manifold")

        return count

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
        if self.cwd_server is not None and hasattr(self.cwd_server, operation):
            op_fn = getattr(self.cwd_server, operation)
            if callable(op_fn):
                # Let exceptions propagate - don't silently fall back to mock
                return cast("dict[str, Any] | list[dict[str, Any]]", op_fn(**params))

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
        elif operation == "deconstruct":
            return {
                "root_id": "mock_root",
                "component_ids": ["mock_1", "mock_2"],
                "message": "Mock decomposition",
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

    def get_metrics(self) -> IntegrationMetrics:
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
