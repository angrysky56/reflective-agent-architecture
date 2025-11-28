"""
LTN Refiner: Logic Tensor Network-based Local Refinement

Provides continuous gradient navigation within and between Hopfield energy basins.
Acts as "topographic handholds" in steep gradient regions where discrete k-NN fails.

Theoretical Foundation:
- Operator C: Belief revision as constrained optimization
- LTNs: Differentiable logic for continuous semantic spaces
- Hybrid Architecture: Combines with RAA's discrete basin hopping

Key Innovation:
Instead of trying to implement full LTNs with complex logic, we use a simplified
"fuzzy constraint" approach where natural language constraints are embedded and
evaluated via cosine similarity. This is computationally lightweight and integrates
naturally with RAA's embedding-based architecture.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn, optim

logger = logging.getLogger(__name__)


@dataclass
class LTNConfig:
    """Configuration for LTN Refiner."""

    # Optimization
    learning_rate: float = 0.01
    max_iterations: int = 100
    convergence_threshold: float = 0.1

    # Loss weights (must sum to 1.0)
    weight_distance: float = 0.2  # L_dist: Minimal change from current
    weight_evidence: float = 0.5  # L_ev: Fit with target evidence
    weight_energy: float = 0.2    # L_energy: Stay in low-energy regions
    weight_constraints: float = 0.1  # L_cons: Satisfy fuzzy constraints

    # Validation thresholds
    max_energy_barrier: float = 5.0  # Maximum energy increase allowed
    min_similarity: float = 0.05     # Minimum movement required (escape basin)
    max_similarity: float = 0.95     # Maximum similarity (must actually move)

    # Device
    device: str = "cpu"

    def __post_init__(self):
        """Validate configuration."""
        total_weight = (
            self.weight_distance +
            self.weight_evidence +
            self.weight_energy +
            self.weight_constraints
        )
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(
                f"Loss weights must sum to 1.0, got {total_weight}. "
                f"Adjust weight_* parameters."
            )


class FuzzyConstraintEvaluator(nn.Module):
    """
    Evaluates natural language constraints via embedding similarity.

    This is a lightweight alternative to full LTN logic evaluation.
    Constraints are embedded once and reused across gradient steps.
    """

    def __init__(self, embedding_fn: callable):
        """
        Initialize evaluator.

        Args:
            embedding_fn: Function that maps text → embedding tensor
        """
        super().__init__()
        self.embedding_fn = embedding_fn
        self.constraint_cache: dict[str, torch.Tensor] = {}

    def embed_constraint(self, constraint: str) -> torch.Tensor:
        """Embed a constraint (with caching)."""
        if constraint not in self.constraint_cache:
            embedding = self.embedding_fn(constraint)
            self.constraint_cache[constraint] = embedding
        return self.constraint_cache[constraint]

    def evaluate(
        self,
        candidate: torch.Tensor,
        constraints: list[str]
    ) -> torch.Tensor:
        """
        Evaluate how well candidate satisfies constraints (fuzzy logic).

        Args:
            candidate: Current candidate embedding
            constraints: List of natural language constraint strings

        Returns:
            loss: Scalar tensor (0 = perfect satisfaction, 1 = complete violation)
        """
        if not constraints:
            return torch.tensor(0.0, device=candidate.device)

        # Compute similarity to each constraint
        violations = []
        for constraint in constraints:
            constraint_emb = self.embed_constraint(constraint).to(candidate.device)

            # Similarity ∈ [-1, 1], map to violation ∈ [0, 1]
            # High similarity (→1) = low violation (→0)
            similarity = F.cosine_similarity(
                candidate.unsqueeze(0),
                constraint_emb.unsqueeze(0),
                dim=1
            )
            violation = (1.0 - similarity) / 2.0  # Maps [1, -1] → [0, 1]
            violations.append(violation)

        # Aggregate violations (mean)
        total_violation = torch.stack(violations).mean()
        return total_violation


class LTNRefiner:
    """
    Local refinement using continuous gradient descent.

    Generates synthetic intermediate waypoints when RAA's discrete search fails.
    Implements Operator C loss function:
        L_total = α·L_dist + β·L_ev + γ·L_energy + δ·L_cons

    Usage:
        refiner = LTNRefiner(embedding_fn, config)
        waypoint = refiner.refine(
            current_belief=belief_embedding,
            evidence=evidence_embedding,
            constraints=["Must be logically consistent"],
            energy_evaluator=manifold.energy
        )
    """

    def __init__(
        self,
        embedding_fn: callable,
        config: Optional[LTNConfig] = None
    ):
        """
        Initialize LTN Refiner.

        Args:
            embedding_fn: Function that maps text → embedding tensor
            config: LTN configuration
        """
        self.config = config or LTNConfig()
        self.embedding_fn = embedding_fn
        self.constraint_evaluator = FuzzyConstraintEvaluator(embedding_fn)
        self.device = torch.device(self.config.device)

        # Statistics for logging
        self.refinement_stats = {
            "total_refinements": 0,
            "successful_refinements": 0,
            "failed_energy_barrier": 0,
            "failed_no_movement": 0,
            "failed_convergence": 0
        }

    def refine(
        self,
        current_belief: torch.Tensor,
        evidence: torch.Tensor,
        constraints: list[str],
        energy_evaluator: callable
    ) -> Optional[torch.Tensor]:
        """
        Generate intermediate waypoint via gradient descent.

        Solves the optimization problem:
            min L_total(x') subject to:
                - E(x') - E(x) < max_energy_barrier (reachable)
                - sim(x', x) < max_similarity (actually moved)
                - sim(x', x) > min_similarity (not too far)

        Args:
            current_belief: Current belief embedding [D]
            evidence: Target evidence embedding [D]
            constraints: Natural language constraint strings
            energy_evaluator: Hopfield energy function

        Returns:
            waypoint: Refined embedding [D] or None if no valid path found
        """
        self.refinement_stats["total_refinements"] += 1

        # Move to device
        current = current_belief.to(self.device)
        target = evidence.to(self.device)

        # Initialize candidate near current belief
        candidate = current.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([candidate], lr=self.config.learning_rate)

        # Store initial energy for barrier check
        with torch.no_grad():
            initial_energy = energy_evaluator(current).item()

        best_loss = float('inf')
        best_candidate = None

        for iteration in range(self.config.max_iterations):
            optimizer.zero_grad()

            # === Loss Components ===

            # L_dist: Minimal change from current belief
            l_dist = torch.norm(candidate - current)

            # L_ev: Evidence fit (minimize distance to target)
            l_ev = 1.0 - F.cosine_similarity(
                candidate.unsqueeze(0),
                target.unsqueeze(0),
                dim=1
            ).squeeze()

            # L_energy: Stay in low-energy regions
            # Note: We normalize by abs(initial_energy) to make scale-invariant
            current_energy = energy_evaluator(candidate)
            l_energy = torch.relu(current_energy - initial_energy) / (
                abs(initial_energy) + 1e-8
            )

            # L_cons: Constraint satisfaction (fuzzy)
            l_cons = self.constraint_evaluator.evaluate(candidate, constraints)

            # === Total Loss (Weighted) ===
            loss = (
                self.config.weight_distance * l_dist +
                self.config.weight_evidence * l_ev +
                self.config.weight_energy * l_energy +
                self.config.weight_constraints * l_cons
            )

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Track best candidate
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_candidate = candidate.detach().clone()

            # === Convergence Check ===
            if loss.item() < self.config.convergence_threshold:
                logger.debug(
                    f"LTN converged at iteration {iteration} "
                    f"(loss={loss.item():.4f})"
                )
                break

        # === Validation ===
        if best_candidate is None:
            self.refinement_stats["failed_convergence"] += 1
            logger.warning("LTN refinement failed to converge")
            return None

        # Use best candidate for validation
        is_valid, reason = self._validate_waypoint(
            candidate=best_candidate,
            current=current,
            energy_evaluator=energy_evaluator,
            initial_energy=initial_energy
        )

        if is_valid:
            self.refinement_stats["successful_refinements"] += 1
            logger.info(
                f"✓ LTN refinement successful (loss={best_loss:.4f}, "
                f"iterations={iteration+1})"
            )
            return best_candidate.cpu()
        else:
            logger.warning(f"LTN waypoint validation failed: {reason}")
            return None

    def _validate_waypoint(
        self,
        candidate: torch.Tensor,
        current: torch.Tensor,
        energy_evaluator: callable,
        initial_energy: float
    ) -> tuple[bool, str]:
        """
        Validate that candidate is a valid waypoint.

        Criteria:
        1. Energy barrier < threshold (reachable)
        2. Actually moved (not stuck in same basin)
        3. Not too far (reasonable step size)

        Returns:
            (is_valid, reason): Validation result and explanation
        """
        with torch.no_grad():
            # Check 1: Energy barrier
            final_energy = energy_evaluator(candidate).item()
            energy_diff = final_energy - initial_energy

            if energy_diff > self.config.max_energy_barrier:
                self.refinement_stats["failed_energy_barrier"] += 1
                return False, (
                    f"Energy barrier too high: ΔE={energy_diff:.2f} > "
                    f"{self.config.max_energy_barrier}"
                )

            # Check 2: Basin separation (must have actually moved)
            similarity = F.cosine_similarity(
                candidate.unsqueeze(0),
                current.unsqueeze(0),
                dim=0
            ).item()

            if similarity > self.config.max_similarity:
                self.refinement_stats["failed_no_movement"] += 1
                return False, (
                    f"Didn't escape basin: similarity={similarity:.4f} > "
                    f"{self.config.max_similarity}"
                )

            if similarity < self.config.min_similarity:
                return False, (
                    f"Moved too far: similarity={similarity:.4f} < "
                    f"{self.config.min_similarity}"
                )

            return True, "Valid waypoint"

    def get_stats(self) -> dict[str, Any]:
        """Return refinement statistics."""
        stats = self.refinement_stats.copy()
        if stats["total_refinements"] > 0:
            stats["success_rate"] = (
                stats["successful_refinements"] / stats["total_refinements"]
            )
        else:
            stats["success_rate"] = 0.0
        return stats

    def reset_stats(self):
        """Reset statistics counters."""
        for key in self.refinement_stats:
            self.refinement_stats[key] = 0


# ============================================================================
# Utility Functions
# ============================================================================


def validate_ltn_config(config: LTNConfig) -> tuple[bool, list[str]]:
    """
    Validate LTN configuration for common issues.

    Returns:
        (is_valid, warnings): Validation result and warning messages
    """
    warnings = []

    # Check loss weights
    if config.weight_evidence < 0.3:
        warnings.append(
            f"weight_evidence={config.weight_evidence} is low. "
            "LTN may not converge toward evidence."
        )

    if config.weight_distance > 0.5:
        warnings.append(
            f"weight_distance={config.weight_distance} is high. "
            "LTN may be too conservative."
        )

    # Check thresholds
    if config.max_energy_barrier < 1.0:
        warnings.append(
            f"max_energy_barrier={config.max_energy_barrier} is very low. "
            "May reject valid waypoints."
        )

    if config.min_similarity > 0.3:
        warnings.append(
            f"min_similarity={config.min_similarity} is high. "
            "May be too restrictive for large conceptual jumps."
        )

    # Check iteration budget
    if config.max_iterations < 50:
        warnings.append(
            f"max_iterations={config.max_iterations} is low. "
            "May fail to converge on complex problems."
        )

    is_valid = len(warnings) == 0
    return is_valid, warnings