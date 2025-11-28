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
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import torch
import torch.nn.functional as f

from .ltn_refiner import LTNConfig, LTNRefiner
from .search_mvp import SearchResult, energy_aware_knn_search
from .sheaf_diagnostics import SheafAnalyzer

logger = logging.getLogger(__name__)


class SearchStrategy(str, Enum):
    """Search strategy used to find result."""
    KNN = "knn"           # RAA discrete k-NN
    LTN = "ltn"           # LTN continuous refinement
    HYBRID = "hybrid"     # Multi-stage approach
    FAILED = "failed"     # All strategies exhausted


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
    enable_ltn_fallback: bool = True      # Use LTN when k-NN fails
    enable_waypoint_storage: bool = True  # Store LTN waypoints in Manifold

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
        manifold,
        ltn_refiner: LTNRefiner,
        sheaf_analyzer: Optional[SheafAnalyzer] = None,
        config: Optional[HybridSearchConfig] = None
    ):
        """
        Initialize hybrid search strategy.

        Args:
            manifold: Manifold (Modern Hopfield Network) for energy evaluation
            ltn_refiner: LTN refiner for continuous navigation
            sheaf_analyzer: Optional Sheaf analyzer for topological validation
            config: Hybrid search configuration
        """
        self.manifold = manifold
        self.ltn = ltn_refiner
        self.sheaf = sheaf_analyzer
        self.config = config or HybridSearchConfig()

        # Statistics
        self.search_stats = {
            "total_searches": 0,
            "knn_success": 0,
            "ltn_success": 0,
            "total_failures": 0,
            "waypoints_stored": 0
        }

    def search(
        self,
        current_state: torch.Tensor,
        evidence: Optional[torch.Tensor] = None,
        constraints: Optional[list[str]] = None,
        context: Optional[dict] = None
    ) -> Optional[HybridSearchResult]:
        """
        Execute hybrid search with automatic strategy selection.

        Args:
            current_state: Current belief/goal embedding [D]
            evidence: Optional target evidence embedding [D]
            constraints: Optional natural language constraints
            context: Optional context dict for logging

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
            ltn_attempted=False
        )

        if self.config.verbose:
            logger.info("=== Hybrid Search Started ===")
            logger.info(f"Evidence provided: {evidence is not None}")
            logger.info(f"Constraints: {len(constraints)}")

        # === Stage 1: Try RAA k-NN Search ===
        knn_result = self._try_knn_search(current_state, result)

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
            current=current_state,
            evidence=evidence,
            constraints=constraints,
            result=result
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
        result: HybridSearchResult
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

        # Execute k-NN search
        try:
            knn_result = energy_aware_knn_search(
                current_state=current_state,
                memory_patterns=memory,
                energy_evaluator=self.manifold.energy,
                k=self.config.knn_k,
                metric=self.config.knn_metric,
                exclude_threshold=self.config.knn_exclude_threshold
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
            if not self._sheaf_validates(knn_result.best_pattern):
                result.knn_failed_reason = "Sheaf validation failed"
                if self.config.verbose:
                    logger.debug("k-NN result failed Sheaf validation")
                return None

        # Convert to HybridSearchResult
        hybrid_result = HybridSearchResult(
            best_pattern=knn_result.best_pattern,
            neighbor_indices=knn_result.neighbor_indices,
            neighbor_distances=knn_result.neighbor_distances,
            selection_score=knn_result.selection_score,
            strategy=SearchStrategy.KNN,
            knn_attempted=True,
            sheaf_validated=self.config.require_sheaf_validation
        )

        return hybrid_result

    def _try_ltn_refinement(
        self,
        current: torch.Tensor,
        evidence: torch.Tensor,
        constraints: list[str],
        result: HybridSearchResult
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
                energy_evaluator=self.manifold.energy
            )
        except Exception as e:
            result.ltn_failed_reason = f"LTN error: {str(e)}"
            logger.warning(result.ltn_failed_reason)
            return None

        if waypoint is None:
            result.ltn_failed_reason = "LTN failed to generate valid waypoint"
            return None

        # Validate with Sheaf (if enabled)
        sheaf_valid = True
        if self.config.require_sheaf_validation and self.sheaf is not None:
            sheaf_valid = self._sheaf_validates(waypoint)
            if not sheaf_valid:
                result.ltn_failed_reason = "Sheaf validation failed"
                if self.config.verbose:
                    logger.debug("LTN waypoint failed Sheaf validation")
                return None

        # Store in Manifold (if enabled)
        stored = False
        if self.config.enable_waypoint_storage:
            try:
                self.manifold.store_pattern(waypoint.unsqueeze(0))
                stored = True
                self.search_stats["waypoints_stored"] += 1
                if self.config.verbose:
                    logger.info("LTN waypoint stored in Manifold for future use")
            except Exception as e:
                logger.warning(f"Failed to store LTN waypoint: {e}")

        # Compute metrics
        energy = self.manifold.energy(waypoint).item()
        distance = f.cosine_similarity(
            waypoint.unsqueeze(0),
            current.unsqueeze(0),
            dim=1
        ).item()

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
            ltn_attempted=True
        )

        return hybrid_result

    def _sheaf_validates(self, pattern: torch.Tensor) -> bool:
        """
        Check topological consistency via Sheaf cohomology.

        Note: Full Sheaf validation requires weight matrices from attention
        mechanisms, which we don't have here. We use a simplified heuristic:
        - Pattern must be in a low-energy region (attractor)

        TODO: Integrate with actual Sheaf diagnostics when attention weights
        are available from the Processor.

        Args:
            pattern: Embedding to validate

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

        if not is_attractor and self.config.verbose:
            logger.debug(f"Pattern not in attractor: E={energy:.4f}")

        # Heuristic 2: Check attention entropy (if supported)
        # High entropy = diffuse attention = confusion/stuck
        is_focused = True
        if hasattr(self.manifold, "get_attention"):
            attention = self.manifold.get_attention(pattern)
            # Entropy = -sum(p * log(p))
            entropy = -torch.sum(attention * torch.log(attention + 1e-10)).item()

            # Max entropy for N patterns is log(N)
            num_patterns = attention.shape[-1]
            if num_patterns > 1:
                max_entropy = torch.log(torch.tensor(float(num_patterns))).item()
                normalized_entropy = entropy / max_entropy

                # If entropy is too high (> 0.8), we are too diffuse
                if normalized_entropy > 0.8:
                    is_focused = False
                    if self.config.verbose:
                        logger.debug(f"High attention entropy: {normalized_entropy:.2f}")

        return is_attractor and is_focused

    def get_stats(self) -> dict[str, Any]:
        """Return search statistics."""
        stats = self.search_stats.copy()

        if stats["total_searches"] > 0:
            stats["knn_success_rate"] = (
                stats["knn_success"] / stats["total_searches"]
            )
            stats["ltn_success_rate"] = (
                stats["ltn_success"] / stats["total_searches"]
            )
            stats["overall_success_rate"] = (
                (stats["knn_success"] + stats["ltn_success"]) /
                stats["total_searches"]
            )
        else:
            stats["knn_success_rate"] = 0.0
            stats["ltn_success_rate"] = 0.0
            stats["overall_success_rate"] = 0.0

        # Add LTN refiner stats
        stats["ltn_refiner_stats"] = self.ltn.get_stats()

        return stats

    def reset_stats(self):
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
    verbose: bool = False
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
    results = {
        "total_cases": len(test_cases),
        "knn_only_success": 0,
        "ltn_rescued": 0,  # Cases where k-NN failed but LTN succeeded
        "both_failed": 0,
        "energy_comparison": [],  # (knn_energy, ltn_energy) pairs
        "distance_comparison": []  # (knn_dist, ltn_dist) pairs
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
                    f"✓ LTN rescued case where k-NN failed "
                    f"(reason: {result.knn_failed_reason})"
                )

    # Calculate success rates
    if results["total_cases"] > 0:
        results["hybrid_success_rate"] = (
            (results["knn_only_success"] + results["ltn_rescued"]) /
            results["total_cases"]
        )
        results["ltn_contribution"] = (
            results["ltn_rescued"] / results["total_cases"]
        )

    return results
