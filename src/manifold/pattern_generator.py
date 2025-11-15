"""
Pattern Generator: Creating Novel Patterns via Composition

Addresses theoretical limitation: Current RAA can only retrieve existing
patterns, not generate truly novel concepts.

This module enables:
- Conceptual blending (interpolation between patterns)
- Pattern composition (combining multiple patterns)
- Exploratory perturbation (stochastic novelty generation)

Theoretical grounding:
- Conceptual Blending Theory (Fauconnier & Turner)
- Analogical reasoning
- Creative insight through recombination
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as f

logger = logging.getLogger(__name__)


class PatternGenerator:
    """
    Generates novel patterns through composition and blending.

    Unlike pure retrieval (k-NN), this enables true creativity:
    conceptual combinations that don't exist in memory.
    """

    def __init__(self, embedding_dim: int, device: str = "cpu"):
        """
        Initialize pattern generator.

        Args:
            embedding_dim: Dimension of pattern embeddings
            device: Computation device
        """
        self.embedding_dim = embedding_dim
        self.device = device

    def blend_patterns(
        self,
        pattern_a: torch.Tensor,
        pattern_b: torch.Tensor,
        blend_weight: float = 0.5,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Conceptual blending via linear interpolation.

        Implements Fauconnier & Turner's conceptual blending:
        create new concept by merging two existing concepts.

        Args:
            pattern_a: First pattern embedding
            pattern_b: Second pattern embedding
            blend_weight: Blending coefficient in [0, 1]
                         0.0 = pure A, 1.0 = pure B, 0.5 = equal blend
            normalize: Whether to normalize result

        Returns:
            Blended pattern embedding
        """
        if pattern_a.shape != pattern_b.shape:
            raise ValueError(
                f"Pattern shapes must match: {pattern_a.shape} vs {pattern_b.shape}"
            )

        # Linear interpolation
        blended = (1 - blend_weight) * pattern_a + blend_weight * pattern_b

        # Normalize to unit sphere (important for Hopfield stability)
        if normalize:
            blended = f.normalize(blended, p=2, dim=-1)

        logger.debug(f"Blended patterns with weight {blend_weight:.2f}")

        return blended

    def compose_patterns(
        self,
        patterns: List[torch.Tensor],
        weights: Optional[List[float]] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Multi-pattern composition via weighted combination.

        Creates novel concept by combining multiple existing patterns.

        Args:
            patterns: List of pattern embeddings
            weights: Optional weight for each pattern (defaults to uniform)
            normalize: Whether to normalize result

        Returns:
            Composed pattern embedding
        """
        if len(patterns) == 0:
            raise ValueError("Must provide at least one pattern")

        if weights is None:
            weights = [1.0 / len(patterns)] * len(patterns)

        if len(weights) != len(patterns):
            raise ValueError("Number of weights must match number of patterns")

        # Weighted sum
        composed = torch.zeros_like(patterns[0])
        for pattern, weight in zip(patterns, weights):
            composed += weight * pattern

        # Normalize
        if normalize:
            composed = f.normalize(composed, p=2, dim=-1)

        logger.debug(f"Composed {len(patterns)} patterns")

        return composed

    def perturb_pattern(
        self,
        pattern: torch.Tensor,
        noise_scale: float = 0.1,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Add random perturbation for exploratory novelty.

        Enables "default mode network" style spontaneous thought:
        small random deviations that might lead to insights.

        Args:
            pattern: Base pattern embedding
            noise_scale: Standard deviation of Gaussian noise
            normalize: Whether to normalize result

        Returns:
            Perturbed pattern embedding
        """
        # Generate random noise
        noise = torch.randn_like(pattern) * noise_scale

        # Add to pattern
        perturbed = pattern + noise

        # Normalize
        if normalize:
            perturbed = f.normalize(perturbed, p=2, dim=-1)

        logger.debug(f"Perturbed pattern with noise scale {noise_scale:.3f}")

        return perturbed

    def analogical_mapping(
        self,
        source_a: torch.Tensor,
        source_b: torch.Tensor,
        target_a: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Analogical reasoning: A is to B as C is to ?

        Computes: target_b = target_a + (source_b - source_a)

        Example: "king" is to "queen" as "man" is to ?
        → "woman" = "man" + ("queen" - "king")

        Args:
            source_a: Source domain pattern A
            source_b: Source domain pattern B
            target_a: Target domain pattern A
            normalize: Whether to normalize result

        Returns:
            Analogically derived pattern B in target domain
        """
        # Compute relation vector in source domain
        relation = source_b - source_a

        # Apply relation to target domain
        target_b = target_a + relation

        # Normalize
        if normalize:
            target_b = f.normalize(target_b, p=2, dim=-1)

        logger.debug("Generated pattern via analogical mapping")

        return target_b

    def spherical_interpolation(
        self,
        pattern_a: torch.Tensor,
        pattern_b: torch.Tensor,
        t: float = 0.5,
    ) -> torch.Tensor:
        """
        Spherical linear interpolation (SLERP).

        Better than linear interpolation for normalized vectors:
        maintains constant magnitude and interpolates along geodesic.

        Args:
            pattern_a: First pattern (assumed normalized)
            pattern_b: Second pattern (assumed normalized)
            t: Interpolation parameter in [0, 1]

        Returns:
            Interpolated pattern on unit sphere
        """
        # Compute angle between patterns
        dot = torch.sum(pattern_a * pattern_b)
        dot = torch.clamp(dot, -1.0, 1.0)  # Numerical stability
        omega = torch.acos(dot)

        # Handle parallel vectors
        if omega.abs() < 1e-6:
            return self.blend_patterns(pattern_a, pattern_b, t, normalize=True)

        # SLERP formula
        sin_omega = torch.sin(omega)
        weight_a = torch.sin((1 - t) * omega) / sin_omega
        weight_b = torch.sin(t * omega) / sin_omega

        interpolated = weight_a * pattern_a + weight_b * pattern_b

        logger.debug(f"SLERP interpolation with t={t:.2f}")

        return interpolated

    def generate_creative_variant(
        self,
        base_patterns: List[torch.Tensor],
        strategy: str = "blend",
        **kwargs
    ) -> torch.Tensor:
        """
        High-level interface for creative pattern generation.

        Args:
            base_patterns: Patterns to use as basis
            strategy: Generation strategy:
                - "blend": Linear blend of two patterns
                - "slerp": Spherical interpolation
                - "compose": Weighted composition
                - "perturb": Add noise to single pattern
                - "analogy": Analogical mapping (requires 3 patterns)
            **kwargs: Strategy-specific parameters

        Returns:
            Generated novel pattern
        """
        if strategy == "blend":
            if len(base_patterns) != 2:
                raise ValueError("Blend requires exactly 2 patterns")
            return self.blend_patterns(
                base_patterns[0],
                base_patterns[1],
                blend_weight=kwargs.get('blend_weight', 0.5),
            )

        elif strategy == "slerp":
            if len(base_patterns) != 2:
                raise ValueError("SLERP requires exactly 2 patterns")
            return self.spherical_interpolation(
                base_patterns[0],
                base_patterns[1],
                t=kwargs.get('t', 0.5),
            )

        elif strategy == "compose":
            return self.compose_patterns(
                base_patterns,
                weights=kwargs.get('weights'),
            )

        elif strategy == "perturb":
            if len(base_patterns) != 1:
                raise ValueError("Perturb requires exactly 1 pattern")
            return self.perturb_pattern(
                base_patterns[0],
                noise_scale=kwargs.get('noise_scale', 0.1),
            )

        elif strategy == "analogy":
            if len(base_patterns) != 3:
                raise ValueError("Analogy requires exactly 3 patterns (A, B, C)")
            return self.analogical_mapping(
                base_patterns[0],
                base_patterns[1],
                base_patterns[2],
            )

        else:
            raise ValueError(f"Unknown generation strategy: {strategy}")


def create_novel_pattern_from_neighbors(
    neighbors: List[torch.Tensor],
    strategy: str = "compose",
    generator: Optional[PatternGenerator] = None,
    **kwargs
) -> torch.Tensor:
    """
    Convenience function: generate novel pattern from k-NN search results.

    Args:
        neighbors: List of neighbor patterns from search
        strategy: Generation strategy
        generator: PatternGenerator instance (creates if None)
        **kwargs: Strategy-specific parameters

    Returns:
        Generated novel pattern
    """
    if len(neighbors) == 0:
        raise ValueError("No neighbors provided")

    if generator is None:
        embedding_dim = neighbors[0].shape[-1]
        generator = PatternGenerator(embedding_dim)

    return generator.generate_creative_variant(neighbors, strategy, **kwargs)
