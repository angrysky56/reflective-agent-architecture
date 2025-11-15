"""
Pattern Memory Management

Utilities for managing and organizing patterns in the Manifold.
"""

from typing import Dict, List, Optional

import torch


class PatternMemory:
    """
    Manages semantic patterns with metadata for the Manifold.

    Provides utilities for:
    - Organizing patterns by category/concept
    - Retrieving patterns with labels
    - Pattern similarity analysis
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.patterns: List[torch.Tensor] = []
        self.labels: List[str] = []
        self.metadata: List[Dict] = []

    def add_pattern(
        self,
        pattern: torch.Tensor,
        label: str,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Add a pattern with label and metadata.

        Args:
            pattern: Embedding vector (embedding_dim,)
            label: Human-readable label for the pattern
            metadata: Optional dictionary with additional information

        Returns:
            Index of the added pattern
        """
        assert pattern.shape[-1] == self.embedding_dim, \
            f"Pattern dimension {pattern.shape[-1]} != {self.embedding_dim}"

        self.patterns.append(pattern.detach().cpu())
        self.labels.append(label)
        self.metadata.append(metadata or {})

        return len(self.patterns) - 1

    def get_pattern(self, index: int) -> torch.Tensor:
        """Get pattern by index."""
        return self.patterns[index]

    def get_label(self, index: int) -> str:
        """Get label by index."""
        return self.labels[index]

    def find_by_label(self, label: str) -> List[int]:
        """Find all pattern indices with matching label."""
        return [i for i, lbl in enumerate(self.labels) if lbl == label]

    def get_all_patterns(self) -> torch.Tensor:
        """
        Get all patterns as a single tensor.

        Returns:
            Tensor of shape (num_patterns, embedding_dim)
        """
        if not self.patterns:
            return torch.empty(0, self.embedding_dim)
        return torch.stack(self.patterns)

    def get_pattern_with_label(self, index: int) -> tuple[torch.Tensor, str]:
        """Get both pattern and label."""
        return self.patterns[index], self.labels[index]

    def find_nearest(
        self,
        query: torch.Tensor,
        k: int = 5,
        metric: str = "cosine"
    ) -> tuple[torch.Tensor, List[int], List[str]]:
        """
        Find k nearest patterns to query.

        Args:
            query: Query embedding (embedding_dim,)
            k: Number of neighbors to return
            metric: Distance metric ("cosine" or "euclidean")

        Returns:
            distances: Distance values
            indices: Pattern indices
            labels: Corresponding labels
        """
        if not self.patterns:
            return torch.empty(0), [], []

        patterns = self.get_all_patterns()

        if metric == "cosine":
            # Cosine similarity (higher = more similar)
            similarities = torch.matmul(
                query.unsqueeze(0),
                patterns.T
            ).squeeze(0)
            distances, indices = torch.topk(similarities, k=min(k, len(patterns)))
            indices = indices.tolist()
        else:  # euclidean
            # Euclidean distance (lower = more similar)
            dists = torch.cdist(query.unsqueeze(0), patterns).squeeze(0)
            distances, indices = torch.topk(dists, k=min(k, len(patterns)), largest=False)
            indices = indices.tolist()

        result_labels = [self.labels[i] for i in indices]

        return distances, indices, result_labels

    def size(self) -> int:
        """Return number of stored patterns."""
        return len(self.patterns)

    def clear(self) -> None:
        """Clear all patterns."""
        self.patterns.clear()
        self.labels.clear()
        self.metadata.clear()

    def __len__(self) -> int:
        return len(self.patterns)

    def __repr__(self) -> str:
        return f"PatternMemory(patterns={len(self.patterns)}, dim={self.embedding_dim})"
