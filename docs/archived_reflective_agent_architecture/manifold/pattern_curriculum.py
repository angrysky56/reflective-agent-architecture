"""
Pattern Curriculum: Strategies for Initializing Manifold Memory

Addresses critical gap: How do patterns get into the Manifold?

The Hopfield network needs semantically meaningful patterns for k-NN
search to work. This module provides strategies for pattern acquisition.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class PatternCurriculumConfig:
    """Configuration for pattern initialization."""

    strategy: str = "random"  # "random", "manual", "pretrained_embeddings", "learned"
    num_initial_patterns: int = 100  # For random/learned initialization
    embedding_dim: int = 512
    random_seed: Optional[int] = 42
    device: str = "cpu"


class PatternCurriculum(ABC):
    """
    Abstract base class for pattern initialization strategies.

    Subclasses define how the Manifold's memory is populated.
    """

    def __init__(self, config: PatternCurriculumConfig):
        self.config = config
        self.device = config.device

    @abstractmethod
    def initialize_patterns(self) -> torch.Tensor:
        """
        Generate initial patterns for the Manifold.

        Returns:
            patterns: Tensor of shape (num_patterns, embedding_dim)
        """
        pass


class RandomPatternCurriculum(PatternCurriculum):
    """
    Random pattern initialization (for testing/debugging).

    Generates random normalized vectors. This is NOT semantically meaningful
    but useful for:
    - Testing Hopfield dynamics
    - Debugging search mechanisms
    - Baseline comparisons
    """

    def initialize_patterns(self) -> torch.Tensor:
        """Generate random normalized patterns."""
        if self.config.random_seed is not None:
            torch.manual_seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

        # Generate random patterns
        patterns = torch.randn(
            self.config.num_initial_patterns,
            self.config.embedding_dim,
            device=self.device,
        )

        # Normalize (important for Hopfield stability)
        patterns = torch.nn.functional.normalize(patterns, p=2, dim=-1)

        logger.info(
            f"Initialized {self.config.num_initial_patterns} random patterns "
            f"(dim={self.config.embedding_dim})"
        )

        return patterns


class ManualPatternCurriculum(PatternCurriculum):
    """
    Manual pattern specification.

    User provides explicit pattern embeddings. Useful for:
    - Controlled experiments
    - Domain-specific knowledge injection
    - Testing specific hypotheses
    """

    def __init__(self, config: PatternCurriculumConfig, patterns: torch.Tensor):
        super().__init__(config)
        self.patterns = patterns

    def initialize_patterns(self) -> torch.Tensor:
        """Return user-provided patterns."""
        # Normalize patterns
        patterns = torch.nn.functional.normalize(self.patterns, p=2, dim=-1)

        logger.info(f"Initialized {patterns.shape[0]} manual patterns")

        return patterns.to(self.device)


class PrototypePatternCurriculum(PatternCurriculum):
    """
    Prototype-based initialization.

    Creates patterns as prototypes of semantic categories.
    For MVP: generates patterns in structured clusters.

    Future: Could use pre-trained embeddings (BERT, GPT) for real semantics.
    """

    def __init__(
        self,
        config: PatternCurriculumConfig,
        num_clusters: int = 10,
        patterns_per_cluster: int = 10,
    ):
        super().__init__(config)
        self.num_clusters = num_clusters
        self.patterns_per_cluster = patterns_per_cluster

    def initialize_patterns(self) -> torch.Tensor:
        """
        Generate clustered patterns.

        Creates semantic structure: patterns within clusters are similar,
        patterns across clusters are dissimilar.
        """
        if self.config.random_seed is not None:
            torch.manual_seed(self.config.random_seed)

        all_patterns = []

        for cluster_idx in range(self.num_clusters):
            # Create cluster centroid
            centroid = torch.randn(self.config.embedding_dim, device=self.device)
            centroid = torch.nn.functional.normalize(centroid, p=2, dim=0)

            # Generate patterns around centroid
            for _ in range(self.patterns_per_cluster):
                # Add small noise to centroid
                noise = torch.randn_like(centroid) * 0.2
                pattern = centroid + noise
                pattern = torch.nn.functional.normalize(pattern, p=2, dim=0)
                all_patterns.append(pattern)

        patterns = torch.stack(all_patterns)

        logger.info(
            f"Initialized {len(all_patterns)} prototype patterns "
            f"({self.num_clusters} clusters × {self.patterns_per_cluster} patterns/cluster)"
        )

        return patterns


def create_pattern_curriculum(config: PatternCurriculumConfig, **kwargs: Any) -> PatternCurriculum:
    """
    Factory function for creating pattern curricula.

    Args:
        config: Configuration
        **kwargs: Strategy-specific arguments

    Returns:
        PatternCurriculum instance
    """
    if config.strategy == "random":
        return RandomPatternCurriculum(config)
    elif config.strategy == "manual":
        if "patterns" not in kwargs:
            raise ValueError("ManualPatternCurriculum requires 'patterns' argument")
        return ManualPatternCurriculum(config, kwargs["patterns"])
    elif config.strategy == "prototype":
        return PrototypePatternCurriculum(
            config,
            num_clusters=kwargs.get("num_clusters", 10),
            patterns_per_cluster=kwargs.get("patterns_per_cluster", 10),
        )
    else:
        raise ValueError(f"Unknown pattern curriculum strategy: {config.strategy}")


# Convenience function for RAA integration
def initialize_manifold_patterns(
    manifold: Any, strategy: str = "prototype", num_patterns: int = 100, **kwargs: Any
) -> int:
    """
    Initialize a Manifold with patterns using specified strategy.

    Args:
        manifold: ModernHopfieldNetwork instance
        strategy: "random", "manual", or "prototype"
        num_patterns: Number of patterns to generate
        **kwargs: Strategy-specific arguments

    Returns:
        Number of patterns added
    """
    config = PatternCurriculumConfig(
        strategy=strategy,
        num_initial_patterns=num_patterns,
        embedding_dim=manifold.embedding_dim,
        device=manifold.device,
    )

    curriculum = create_pattern_curriculum(config, **kwargs)
    patterns = curriculum.initialize_patterns()

    # Store patterns in manifold
    for pattern in patterns:
        manifold.store_pattern(pattern)

    logger.info(f"Manifold initialized with {patterns.shape[0]} patterns")

    return int(patterns.shape[0])
