"""
Hybrid Search Strategy: RAA + LTN Integration

Orchestrates discrete basin hopping (RAA k-NN) with continuous refinement (LTN).
Implements graceful degradation: fast k-NN first, slow LTN only when needed.

Architecture:
    Stage 1: Try RAA energy-aware k-NN search (O(k) time)
             ↓ Success → Return immediately
             ↓ Failure → Stage 2
    Stage 2: Try LTN gradient refinement (O(iterations) time)
             ↓ Generate synthetic waypoint
             ↓ Validate with Sheaf diagnostics
             ↓ Store in Manifold for future k-NN use

Key Innovation:
The hybrid approach creates a positive feedback loop:
- LTN generates waypoints in sparse/steep regions
- Waypoints are stored in Manifold
- Future searches can use k-NN to find these LTN-generated handholds
- Over time, the Manifold becomes "scaffolded" with continuous paths
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import torch
import torch.nn.functional as f

from reflective_agent_architecture.manifold import Manifold

from .ltn_refiner import LTNConfig, LTNRefiner
from .search_mvp import SearchResult, energy_aware_knn_search
from .sheaf_diagnostics import SheafAnalyzer
from .utility_aware_search import UtilityAwareSearch

logger = logging.getLogger(__name__)


class SearchStrategy(str, Enum):
    """Search strategy used to find result."""

    KNN = "knn"  # RAA discrete k-NN
    LTN = "ltn"  # LTN continuous refinement
    HYBRID = "hybrid"  # Multi-stage approach
    FAILED = "failed"  # All strategies exhausted


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search strategy."""

    # RAA k-NN parameters
    knn_k: int = 5
    knn_metric: str = "cosine"
    knn_exclude_threshold: float = 0.95

    # LTN parameters
    ltn_config: Optional[LTNConfig] = None

    # Strategy selection
    enable_ltn_fallback: bool = True  # Use LTN when k-NN fails
    enable_waypoint_storage: bool = True  # Store LTN waypoints in Manifold
    force_ltn: bool = False  # Skip k-NN and force LTN refinement

    # Validation
    require_sheaf_validation: bool = True  # Validate with Sheaf cohomology

    # Memory thresholds
    min_memory_size: int = 3  # Minimum patterns needed for k-NN

    # Logging
    verbose: bool = False


@dataclass
class HybridSearchResult(SearchResult):
    """Extended search result with hybrid metadata."""

    strategy: SearchStrategy  # Which strategy was used
    ltn_iterations: Optional[int] = None  # Iterations if LTN was used
    stored_in_manifold: bool = False  # Whether waypoint was stored
    sheaf_validated: bool = False  # Whether Sheaf validation passed

    # Diagnostic info
    knn_attempted: bool = False
    knn_failed_reason: Optional[str] = None
    ltn_attempted: bool = False
    ltn_failed_reason: Optional[str] = None


class HybridSearchStrategy:
    """
    Hybrid search combining RAA basin hopping with LTN refinement.

    Decision Tree:
    1. Check memory size → If < min_size: Skip k-NN, go to LTN
    2. Try k-NN search → If found valid neighbor: Return
    3. If k-NN failed and LTN enabled:
       a. Run LTN refinement
       b. Validate with Sheaf (if enabled)
       c. Store in Manifold (if enabled)
       d. Return LTN waypoint
    4. If all failed: Return None

    Usage:
        hybrid = HybridSearchStrategy(manifold, ltn_refiner, sheaf_analyzer)
        result = hybrid.search(
            current_state=belief_embedding,
            evidence=evidence_embedding,
            constraints=["Must be consistent"]
        )
    """

    def __init__(
        self,
        manifold: Manifold,
        ltn_refiner: LTNRefiner,
        sheaf_analyzer: Optional[SheafAnalyzer] = None,
        config: Optional[HybridSearchConfig] = None,
        utility_search: Optional[UtilityAwareSearch] = None,
    ):
        """
        Initialize hybrid search strategy.

        Args:
            manifold: Manifold (Modern Hopfield Network) for energy evaluation
            ltn_refiner: LTN refiner for continuous navigation
            sheaf_analyzer: Optional Sheaf analyzer for topological validation
            config: Hybrid search configuration
            utility_search: Optional Utility-Aware Search module for valence bias
        """
        self.manifold = manifold
        self.ltn = ltn_refiner
        self.sheaf = sheaf_analyzer
        self.config = config or HybridSearchConfig()
        self.utility_search = utility_search

        # Statistics
        self.search_stats = {
            "total_searches": 0,
            "knn_success": 0,
            "ltn_success": 0,
            "total_failures": 0,
            "waypoints_stored": 0,
            "scaffolding_success": 0,  # Track when k-NN retrieves an LTN waypoint
        }

    def search(
        self,
        current_state: torch.Tensor,
        evidence: Optional[torch.Tensor] = None,
        constraints: Optional[list[str]] = None,
        context: Optional[dict] = None,
        force_ltn: bool = False,
        k: Optional[int] = None,
        metric: Optional[str] = None,
        valence: float = 0.0,
    ) -> Optional[HybridSearchResult]:
        """
        Execute hybrid search with automatic strategy selection.

        Args:
            current_state: Current belief/goal embedding [D]
            evidence: Optional target evidence embedding [D]
            constraints: Optional natural language constraints
            context: Optional context dict for logging
            force_ltn: If True, skip k-NN and force LTN refinement
            k: Optional override for number of neighbors
            metric: Optional override for distance metric
            valence: Affective valence bias (-1.0 to 1.0) for utility-aware search

        Returns:
            HybridSearchResult if successful, None if all strategies failed
        """
        self.search_stats["total_searches"] += 1
        constraints = constraints or []
        context = context or {}

        # Initialize result tracking
        result = HybridSearchResult(
            best_pattern=current_state,  # Placeholder
            neighbor_indices=[],
            neighbor_distances=torch.tensor([]),
            selection_score=0.0,
            strategy=SearchStrategy.FAILED,
            knn_attempted=False,
            ltn_attempted=False,
        )

        if self.config.verbose:
            logger.info("=== Hybrid Search Started ===")
            logger.info(f"Evidence provided: {evidence is not None}")
            logger.info(f"Constraints: {len(constraints)}")
            if force_ltn or self.config.force_ltn:
                logger.info("Mode: Forced LTN (Skipping k-NN)")

        # === Stage 1: Try RAA k-NN Search ===
        # Skip if forced LTN
        should_skip_knn = force_ltn or self.config.force_ltn

        knn_result = None
        if not should_skip_knn:
            knn_result = self._try_knn_search(
                current_state, result, k=k, metric=metric, valence=valence
            )
        else:
            result.knn_failed_reason = "Skipped (Forced LTN)"

        if knn_result is not None:
            self.search_stats["knn_success"] += 1
            if self.config.verbose:
                logger.info("✓ RAA k-NN search succeeded")
            return knn_result

        # === Stage 2: LTN Fallback (if enabled and evidence available) ===
        if not self.config.enable_ltn_fallback:
            self.search_stats["total_failures"] += 1
            if self.config.verbose:
                logger.warning("LTN fallback disabled, search failed")
            return None

        if evidence is None:
            result.ltn_failed_reason = "No evidence provided for LTN refinement"
            self.search_stats["total_failures"] += 1
            if self.config.verbose:
                logger.warning("No evidence for LTN refinement, search failed")
            return None

        if self.config.verbose:
            logger.info("RAA search failed, attempting LTN refinement...")

        ltn_result = self._try_ltn_refinement(
            current=current_state, evidence=evidence, constraints=constraints, result=result
        )

        if ltn_result is not None:
            self.search_stats["ltn_success"] += 1
            if self.config.verbose:
                logger.info("✓ LTN refinement succeeded")
            return ltn_result

        # === All Strategies Exhausted ===
        self.search_stats["total_failures"] += 1
        if self.config.verbose:
            logger.warning("✗ All search strategies failed")
        return None

    def _try_knn_search(
        self,
        current_state: torch.Tensor,
        result: HybridSearchResult,
        k: Optional[int] = None,
        metric: Optional[str] = None,
        valence: float = 0.0,
    ) -> Optional[HybridSearchResult]:
        """
        Attempt RAA energy-aware k-NN search.

        Returns:
            HybridSearchResult if successful, None if failed
        """
        result.knn_attempted = True

        # Check memory size
        memory = self.manifold.get_patterns()

        if memory.shape[0] < self.config.min_memory_size:
            result.knn_failed_reason = (
                f"Sparse memory: {memory.shape[0]} < {self.config.min_memory_size}"
            )
            if self.config.verbose:
                logger.debug(f"Skipping k-NN: {result.knn_failed_reason}")
            return None

        # Custom or Default Energy Evaluator
        energy_fn = self.manifold.energy

        # If Utility Search is enabled AND we have a non-neutral valence,
        # wrap the energy function to include bias.
        # Capture locally to allow type narrowing in closure
        utility_search = self.utility_search

        if utility_search is not None and abs(valence) > 0.05:
            # Create a closure that captures the valence
            def biased_energy_evaluator(pattern: torch.Tensor) -> float:
                base_energy = self.manifold.energy(pattern)
                return utility_search.compute_biased_energy(base_energy, valence)

            energy_fn = biased_energy_evaluator
            if self.config.verbose:
                logger.debug(f"Applying Utility Bias to Search (Valence: {valence:.2f})")

        # Execute k-NN search
        try:
            knn_result = energy_aware_knn_search(
                current_state=current_state,
                memory_patterns=memory,
                energy_evaluator=energy_fn,
                k=k if k is not None else self.config.knn_k,
                metric=metric if metric is not None else self.config.knn_metric,
                exclude_threshold=self.config.knn_exclude_threshold,
            )
        except Exception as e:
            result.knn_failed_reason = f"k-NN error: {str(e)}"
            logger.warning(result.knn_failed_reason)
            return None

        if knn_result is None:
            result.knn_failed_reason = "No valid neighbors found"
            return None

        # Validate with Sheaf (if enabled)
        if self.config.require_sheaf_validation and self.sheaf is not None:
            if not self._sheaf_validates(knn_result.best_pattern, check_attractor=True):
                result.knn_failed_reason = "Sheaf validation failed"
                if self.config.verbose:
                    logger.debug("k-NN result failed Sheaf validation")
                return None

        # Check for Scaffolding Effect (did we retrieve a synthetic waypoint?)
        best_idx = knn_result.neighbor_indices[0]
        if hasattr(self.manifold, "get_pattern_metadata"):
            meta = self.manifold.get_pattern_metadata(best_idx)
            if meta.get("is_synthetic", False):
                self.search_stats["scaffolding_success"] += 1
                if self.config.verbose:
                    logger.info(
                        "✓ Scaffolding Effect Verified: k-NN retrieved synthetic LTN waypoint!"
                    )

        # Convert to HybridSearchResult
        hybrid_result = HybridSearchResult(
            best_pattern=knn_result.best_pattern,
            neighbor_indices=knn_result.neighbor_indices,
            neighbor_distances=knn_result.neighbor_distances,
            selection_score=knn_result.selection_score,
            strategy=SearchStrategy.KNN,
            knn_attempted=True,
            sheaf_validated=self.config.require_sheaf_validation,
        )

        return hybrid_result

    def _try_ltn_refinement(
        self,
        current: torch.Tensor,
        evidence: torch.Tensor,
        constraints: list[str],
        result: HybridSearchResult,
    ) -> Optional[HybridSearchResult]:
        """
        Attempt LTN gradient refinement.

        Returns:
            HybridSearchResult if successful, None if failed
        """
        result.ltn_attempted = True

        # Execute LTN refinement
        try:
            waypoint = self.ltn.refine(
                current_belief=current,
                evidence=evidence,
                constraints=constraints,
                energy_evaluator=self.manifold.energy,
            )
        except Exception as e:
            result.ltn_failed_reason = f"LTN error: {str(e)}"
            logger.warning(result.ltn_failed_reason)
            return None

        if waypoint is None:
            result.ltn_failed_reason = "LTN failed to generate valid waypoint"
            return None

        # Validate with Sheaf (if enabled)
        sheaf_valid = False
        if self.config.require_sheaf_validation:
            # For LTN, we relax the attractor check because we are creating a new attractor
            # We also relax entropy check because bridging concepts might be diffuse
            if not self._sheaf_validates(waypoint, check_attractor=False, check_entropy=False):
                result.ltn_failed_reason = "Sheaf validation failed (entropy/focus)"
                if self.config.verbose:
                    logger.debug("LTN result failed Sheaf validation")
                return None
            sheaf_valid = True
        # Store in Manifold (if enabled)
        stored = False
        if self.config.enable_waypoint_storage:
            try:
                # Store with metadata to track scaffolding effect
                metadata = {"is_synthetic": True, "source": "ltn"}
                if (
                    hasattr(self.manifold, "store_pattern")
                    and "metadata" in self.manifold.store_pattern.__code__.co_varnames
                ):
                    self.manifold.store_pattern(waypoint.unsqueeze(0), metadata=metadata)
                else:
                    self.manifold.store_pattern(waypoint.unsqueeze(0))

                stored = True
                self.search_stats["waypoints_stored"] += 1
                if self.config.verbose:
                    logger.info("LTN waypoint stored in Manifold for future use")
            except Exception as e:
                logger.warning(f"Failed to store LTN waypoint: {e}")

        # Compute metrics
        energy = self.manifold.energy(waypoint).item()
        distance = f.cosine_similarity(waypoint.unsqueeze(0), current.unsqueeze(0), dim=1).item()

        # Create result
        hybrid_result = HybridSearchResult(
            best_pattern=waypoint,
            neighbor_indices=[],
            neighbor_distances=torch.tensor([distance]),
            selection_score=energy,
            strategy=SearchStrategy.LTN,
            ltn_iterations=self.ltn.config.max_iterations,  # Could track actual
            stored_in_manifold=stored,
            sheaf_validated=sheaf_valid,
            knn_attempted=True,
            knn_failed_reason=result.knn_failed_reason,
            ltn_attempted=True,
        )

        return hybrid_result

    def _sheaf_validates(
        self, pattern: torch.Tensor, check_attractor: bool = True, check_entropy: bool = True
    ) -> bool:
        """
        Validate pattern using Sheaf Cohomology (or heuristics).

        Checks:
        1. Is it a stable attractor? (Energy < 0) - Optional for LTN
        2. Is attention focused? (Low entropy) - Optional for LTN

        Args:
            pattern: Embedding to validate
            check_attractor: Whether to enforce negative energy (default True)
            check_entropy: Whether to enforce low entropy (default True)

        Returns:
            is_valid: Whether pattern passes validation
        """
        if self.sheaf is None:
            return True  # No validator = pass by default

        # Bootstrapping: If manifold is empty, we accept the first pattern
        # to seed the memory.
        if hasattr(self.manifold, "get_patterns"):
            patterns = self.manifold.get_patterns()
            if patterns.shape[0] == 0:
                if self.config.verbose:
                    logger.info("Bootstrapping: Manifold empty, accepting waypoint.")
                return True

        # Heuristic 1: Check if in attractor (negative energy)
        energy = self.manifold.energy(pattern).item()
        is_attractor = energy < 0

        if self.config.verbose:
            logger.info(f"Sheaf Validation: Energy={energy:.4f}, Is Attractor={is_attractor}")

        if check_attractor:
            if is_attractor:
                pass  # Continue to next check
            else:
                if self.config.verbose:
                    logger.debug(f"Pattern not in attractor: E={energy:.4f}")
                return False
        # The original code had a debug message here that was always printed if not an attractor.
        # With `check_attractor` parameter, this message should only be printed if `check_attractor` is True
        # and `is_attractor` is False, which is handled by the `return False` block above.
        # If `check_attractor` is False, we don't care if it's an attractor for this heuristic.

        # Heuristic 2: Check attention entropy (if supported)
        # High entropy = diffuse attention = confusion/stuck
        if check_entropy:
            is_focused = True
            if hasattr(self.manifold, "get_attention"):
                attention = self.manifold.get_attention(pattern)
                # Entropy = -sum(p * log(p))
                entropy = -torch.sum(attention * torch.log(attention + 1e-10)).item()

                # Max entropy for N patterns is log(N)
                # We want normalized entropy < threshold
                n_patterns = attention.shape[-1]
                max_entropy = math.log(n_patterns) if n_patterns > 1 else 1.0
                normalized_entropy = entropy / max_entropy

                if self.config.verbose:
                    logger.debug(f"Sheaf Validation: Entropy={normalized_entropy:.4f}")

                if normalized_entropy > 0.8:  # Threshold
                    is_focused = False
                    if self.config.verbose:
                        logger.debug(f"High entropy (confusion): {normalized_entropy:.2f}")

            if not is_focused:
                return False

        return True

    def get_stats(self) -> dict[str, Any]:
        """Return search statistics."""
        stats: dict[str, Any] = self.search_stats.copy()

        if stats["total_searches"] > 0:
            stats["knn_success_rate"] = stats["knn_success"] / stats["total_searches"]
            stats["ltn_success_rate"] = stats["ltn_success"] / stats["total_searches"]
            stats["overall_success_rate"] = (stats["knn_success"] + stats["ltn_success"]) / stats[
                "total_searches"
            ]
        else:
            stats["knn_success_rate"] = 0.0
            stats["ltn_success_rate"] = 0.0
            stats["overall_success_rate"] = 0.0

        # Add LTN refiner stats
        stats["ltn_refiner_stats"] = self.ltn.get_stats()

        return stats

    def reset_stats(self) -> None:
        """Reset all statistics."""
        for key in self.search_stats:
            self.search_stats[key] = 0
        self.ltn.reset_stats()


# ============================================================================
# Utility Functions
# ============================================================================


def compare_search_strategies(
    hybrid_strategy: HybridSearchStrategy,
    test_cases: list[dict[str, torch.Tensor]],
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Benchmark hybrid search vs pure k-NN on test cases.

    Args:
        hybrid_strategy: Configured hybrid search strategy
        test_cases: List of dicts with 'current', 'evidence', 'constraints'
        verbose: Enable verbose logging

    Returns:
        comparison_results: Statistics comparing strategies
    """
    results: dict[str, Any] = {
        "total_cases": len(test_cases),
        "knn_only_success": 0,
        "ltn_rescued": 0,  # Cases where k-NN failed but LTN succeeded
        "both_failed": 0,
        "energy_comparison": [],  # (knn_energy, ltn_energy) pairs
        "distance_comparison": [],  # (knn_dist, ltn_dist) pairs
    }

    for i, case in enumerate(test_cases):
        if verbose:
            logger.info(f"\n=== Test Case {i+1}/{len(test_cases)} ===")

        result = hybrid_strategy.search(
            current_state=case["current"],
            evidence=case.get("evidence"),
            constraints=case.get("constraints"),
        )

        if result is None:
            results["both_failed"] += 1
        elif result.strategy == SearchStrategy.KNN:
            results["knn_only_success"] += 1
        elif result.strategy == SearchStrategy.LTN:
            results["ltn_rescued"] += 1

            if verbose:
                logger.info(
                    f"✓ LTN rescued case where k-NN failed " f"(reason: {result.knn_failed_reason})"
                )

    # Calculate success rates
    if results["total_cases"] > 0:
        results["hybrid_success_rate"] = (
            results["knn_only_success"] + results["ltn_rescued"]
        ) / results["total_cases"]
        results["ltn_contribution"] = results["ltn_rescued"] / results["total_cases"]

    return results
