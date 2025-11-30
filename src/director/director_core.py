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
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import torch

from src.compass.adapters import RAALLMProvider
from src.compass.compass_framework import COMPASS

if TYPE_CHECKING:
    from src.integration.continuity_service import ContinuityService

from .entropy_monitor import EntropyMonitor
from .hybrid_search import HybridSearchConfig, HybridSearchStrategy
from .ltn_refiner import LTNConfig, LTNRefiner
from .matrix_monitor import MatrixMonitor, MatrixMonitorConfig
from .search_mvp import SearchResult
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
    device: str = "cpu"

    # Sheaf Diagnostics
    sheaf_config: Optional[SheafConfig] = None

    # Matrix Monitor
    matrix_monitor_config: Optional[MatrixMonitorConfig] = None

    # Hybrid Search
    hybrid_search_config: Optional[HybridSearchConfig] = None


class DirectorMVP:
    """
    Director MVP: Metacognitive Monitor + Search Engine.

    The Director is the core innovation of RAA. It detects confusion via
    entropy monitoring and triggers search in the Manifold for alternative
    conceptual framings.
    """

    def __init__(
        self,
        manifold,
        config: Optional[DirectorConfig] = None,
        embedding_fn: Optional[Callable[[str], torch.Tensor]] = None,
        mcp_client: Optional[Any] = None,
        continuity_service: Optional["ContinuityService"] = None,
        llm_provider: Optional[Any] = None,
    ):
        """
        Initialize Director.

        Args:
            manifold: Manifold (Modern Hopfield Network) to search
            config: Director configuration
            embedding_fn: Function to embed text for LTN constraints
            mcp_client: Optional MCP client for tool execution
            continuity_service: Optional ContinuityService for anchoring milestones
        """
        self.manifold = manifold
        self.config = config or DirectorConfig()
        self.mcp_client = mcp_client
        self.continuity_service = continuity_service

        # Entropy monitor
        self.monitor = EntropyMonitor(
            threshold_percentile=self.config.entropy_threshold_percentile,
            history_size=self.config.entropy_history_size,
            default_threshold=self.config.default_entropy_threshold,
        )

        # Sheaf Analyzer
        sheaf_config = self.config.sheaf_config or SheafConfig(device=self.config.device)
        self.sheaf_analyzer = SheafAnalyzer(sheaf_config)

        # 3. Matrix Monitor (Cognitive Proprioception)
        self.matrix_monitor = MatrixMonitor(
            config=self.config.matrix_monitor_config or MatrixMonitorConfig(device=self.config.device)
        )
        self.matrix_monitor.seed_defaults()

        # 4. Hybrid Search Strategy (Operator C)
        hybrid_cfg = self.config.hybrid_search_config or HybridSearchConfig(
            knn_k=self.config.search_k,
            knn_metric=self.config.search_metric,
            knn_exclude_threshold=self.config.exclude_threshold,
        )

        # Initialize LTN Refiner
        # Use provided embedding_fn or a dummy that warns
        def dummy_embed(text: str) -> torch.Tensor:
            logger.warning("Using dummy embedding function. Constraints will fail.")
            # Return a 1D tensor of size 1 just to satisfy type checks,
            # though it will likely fail dimension checks if used.
            # Ideally we should raise an error if constraints are used.
            return torch.zeros(1, device=self.config.device)

        ltn_config = hybrid_cfg.ltn_config or LTNConfig(device=self.config.device)
        self.ltn_refiner = LTNRefiner(
            embedding_fn=embedding_fn or dummy_embed,
            config=ltn_config
        )

        self.hybrid_search = HybridSearchStrategy(
            manifold=self.manifold,
            ltn_refiner=self.ltn_refiner,
            sheaf_analyzer=self.sheaf_analyzer,
            config=hybrid_cfg
        )

        # 5. COMPASS Framework (Metacognitive Orchestration)
        # Initialize with RAA LLM adapter and MCP client
        # Use provided llm_provider or create default
        _llm_provider = llm_provider if llm_provider is not None else RAALLMProvider()
        self.compass = COMPASS(llm_provider=_llm_provider, mcp_client=self.mcp_client)

        # Cognitive State
        self.latest_cognitive_state: tuple[str, float] = ("Unknown", 0.0)
        self.latest_diagnostics: dict[str, Any] = {}
        self.latest_attention_weights: Optional[torch.Tensor] = None

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
        # Delegate to Hybrid Search Strategy
        # This handles both fast k-NN and slow LTN refinement
        try:
            result = self.hybrid_search.search(
                current_state=current_state,
                evidence=None, # Director search is usually unsupervised/intrinsic, unless context provides evidence
                constraints=[],
                context=context
            )

            # Log search episode
            if self.config.log_search_episodes and result:
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
        Main Director loop: monitor -> detect -> compute adaptive beta -> search -> return.

        This is the primary interface used by the RAA integration.

        The Director now dynamically adjusts exploration based on confusion:
        - High entropy (confusion) -> low beta -> more exploratory search
        - Low entropy (confidence) -> high beta -> more focused search

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

        # --- COMPASS Resource Allocation (oMCD) ---
        # Map Director state to oMCD state
        # High entropy = Low value difference (hard decision)
        # Precision = 1/variance (approximate)

        # Heuristic mapping:
        # value_difference ~ 1 / (1 + entropy)
        # precision ~ 1.0 (default)

        omcd_state = {
            "value_difference": 1.0 / (1.0 + entropy_value),
            "precision": 1.0,
            "variance": 1.0
        }

        # Determine allocation
        allocation = self.compass.omcd_controller.determine_resource_allocation(
            current_state=omcd_state,
            importance=10.0, # Default importance
            available_resources=100.0 # Default available
        )

        logger.info(f"COMPASS oMCD Allocation: {allocation['amount']:.2f} resources (Confidence: {allocation['confidence']:.3f})")
        context["compass_allocation"] = allocation

        # If allocation is very high, we might want to trigger full COMPASS processing
        # For now, we just log it and use it to inform search (potentially)
        if allocation['amount'] > 80.0:
            logger.warning(f"High entropy/allocation detected ({allocation['amount']:.2f}). Triggering COMPASS intervention.")

            # Trigger COMPASS asynchronously
            import asyncio
            try:
                # We need to get the running loop or create a task
                # Since check_and_search is sync, we assume there's an event loop running (e.g. in the server)
                loop = asyncio.get_running_loop()

                task_desc = f"High Entropy Intervention: Entropy={entropy_value:.2f}, Allocation={allocation['amount']:.2f}. Analyze context and intervene."
                # Pass current context copy
                task_ctx = context.copy() if context else {}

                loop.create_task(self.compass.process_task(task_desc, task_ctx))
                logger.info("COMPASS intervention task scheduled.")

            except RuntimeError:
                logger.warning("No running event loop. Skipping async COMPASS trigger.")
            except Exception as e:
                logger.error(f"Failed to trigger COMPASS: {e}")

        # ------------------------------------------

        # Step 2: Check if intervention needed and perform search
        new_goal = None
        if is_clash:
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

            # Anchor this milestone (Continuity)
            if self.continuity_service:
                try:
                    # Convert tensor to numpy for continuity service
                    state_np = new_goal.cpu().numpy() if isinstance(new_goal, torch.Tensor) else new_goal
                    # Ensure it's 1D or flatten it
                    if len(state_np.shape) > 1:
                        state_np = state_np.flatten()

                    self.continuity_service.add_anchor(
                        agent_id="system",
                        vector=state_np,
                        metadata={
                            "type": "intervention",
                            "trigger": "clash",
                            "entropy": float(entropy_value),
                            "source": "director_search"
                        }
                    )
                    logger.info("Anchored new goal state in Continuity Field.")
                except Exception as e:
                    logger.warning(f"Failed to anchor state: {e}")
        else:
            logger.debug(f"No clash detected (entropy={entropy_value:.3f})")
            return None


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

    def monitor_thought_process(self, attention_weights: torch.Tensor) -> tuple[str, float]:
        """
        Monitor the 'topology' of thought by analyzing attention matrices.
        Returns the cognitive state label and its energy (stability).
        """
        # Store raw weights for potential feedback/teaching
        self.latest_attention_weights = attention_weights

        state, energy, diagnostics = self.matrix_monitor.check_state(attention_weights)
        self.latest_cognitive_state = (state, energy)
        self.latest_diagnostics = diagnostics
        logger.debug(f"Director monitored thought process. State: {state}")
        return state, energy

    def teach_state(self, label: str) -> bool:
        """
        Teach the Director that the *last* monitored thought corresponds to 'label'.
        """
        if self.latest_attention_weights is None:
            logger.warning("Cannot teach state: No recent thought process recorded.")
            return False

        self.matrix_monitor.register_state(self.latest_attention_weights, label)
        logger.info(f"Director learned new state: {label}")
        return True

    def get_known_states(self) -> Dict[int, str]:
        """List all known cognitive states."""
        return self.matrix_monitor.state_labels.copy()

    def visualize_last_thought(self) -> str:
        """Get ASCII visualization of the last thought."""
        if self.latest_attention_weights is None:
            return "No recent thought to visualize."
        return self.matrix_monitor.visualize_topology(self.latest_attention_weights)

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
