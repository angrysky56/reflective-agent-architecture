"""
Search Mechanism MVP: k-Nearest Neighbors

Phase 1 implementation of the Director's search mechanism.
Uses k-NN in embedding space to find alternative goal framings.

This is the simplest approach that:
- Gets the prototype running quickly
- Is interpretable (can inspect retrieved neighbors)
- Works without training
- Enables rapid iteration

Based on SEARCH_MECHANISM_DESIGN.md Phase 1 specification.
"""

from dataclasses import dataclass
from typing import Callable, List

import torch
import torch.nn.functional as f


@dataclass
class SearchResult:
    """Result from search operation."""

    best_pattern: torch.Tensor  # Selected alternative pattern
    neighbor_indices: List[int]  # Indices of k nearest patterns
    neighbor_distances: torch.Tensor  # Distances to neighbors
    selection_score: float  # Score used for selection (e.g., entropy reduction)


def knn_search(
    current_state: torch.Tensor,
    memory_patterns: torch.Tensor,
    k: int = 5,
    metric: str = "cosine",
    exclude_threshold: float = 0.95,
) -> SearchResult:
    """
    find k nearest neighbor patterns in Hopfield memory.

    Implements Phase 1 search mechanism: retrieval-based search using
    geometric proximity as proxy for semantic similarity.

    Args:
        current_state: Current embedding (embedding_dim,) or (batch, embedding_dim)
        memory_patterns: Stored patterns (num_patterns, embedding_dim)
        k: Number of neighbors to retrieve
        metric: Distance metric ("cosine" or "euclidean")
        exclude_threshold: Similarity threshold for excluding current basin

    Returns:
        SearchResult with best pattern and neighbor information
    """
    if current_state.dim() == 1:
        current_state = current_state.unsqueeze(0)  # (1, embedding_dim)

    num_patterns = memory_patterns.shape[0]

    if num_patterns == 0:
        raise ValueError("Cannot search in empty memory")

    # Ensure k doesn't exceed available patterns
    k = min(k, num_patterns)

    # Compute distances based on metric
    if metric == "cosine":
        # Normalize vectors for cosine similarity
        current_norm = f.normalize(current_state, p=2, dim=-1)
        patterns_norm = f.normalize(memory_patterns, p=2, dim=-1)

        # Cosine similarity (higher = more similar)
        similarities = torch.matmul(current_norm, patterns_norm.T).squeeze(0)

        # Exclude patterns too similar to current (same basin)
        mask = similarities < exclude_threshold
        similarities = similarities * mask.float() + (1 - mask.float()) * -1e9

        # Get top-k
        distances, indices = torch.topk(similarities, k=k, largest=True)

    else:  # euclidean
        # Euclidean distance (lower = more similar)
        dists = torch.cdist(current_state, memory_patterns).squeeze(0)

        # Exclude very close patterns (same basin)
        min_dist = dists.min() * exclude_threshold
        mask = dists > min_dist
        dists = dists * mask.float() + (1 - mask.float()) * 1e9

        # Get top-k (smallest distances)
        distances, indices = torch.topk(dists, k=k, largest=False)

    # Convert to lists
    neighbor_indices = indices.tolist()

    # Selection strategy: for MVP, just take nearest neighbor
    # TODO: In Phase 2, select based on entropy reduction when used as goal
    best_pattern = memory_patterns[neighbor_indices[0]]

    # Selection score (for logging/debugging)
    selection_score = distances[0].item()

    result = SearchResult(
        best_pattern=best_pattern,
        neighbor_indices=neighbor_indices,
        neighbor_distances=distances,
        selection_score=selection_score,
    )

    return result


def energy_aware_knn_search(
    current_state: torch.Tensor,
    memory_patterns: torch.Tensor,
    energy_evaluator: Callable[[torch.Tensor], torch.Tensor],
    k: int = 5,
    metric: str = "cosine",
    exclude_threshold: float = 0.95,
) -> SearchResult:
    """
    Energy-aware k-NN search aligned with Hopfield principles.

    This addresses the critical gap: geometric proximity alone doesn't
    guarantee the pattern is a stable attractor. We must check energy!

    Process:
    1. Get k nearest neighbors by geometric distance
    2. Evaluate Hopfield energy for each neighbor
    3. Select pattern with LOWEST energy (most stable attractor)

    Args:
        current_state: Current embedding
        memory_patterns: Stored patterns
        energy_evaluator: Function computing Hopfield energy
                         Signature: (pattern: Tensor) -> Tensor (scalar)
        k: Number of neighbors to retrieve
        metric: Distance metric for initial retrieval
        exclude_threshold: Similarity threshold for excluding current basin

    Returns:
        SearchResult with lowest-energy (most stable) pattern
    """
    # Step 1: Get k nearest neighbors geometrically
    basic_result = knn_search(current_state, memory_patterns, k, metric, exclude_threshold)

    # Step 2: Evaluate Hopfield energy for each neighbor
    best_pattern = None
    best_energy = float("inf")

    for idx in basic_result.neighbor_indices:
        pattern = memory_patterns[idx]
        energy = energy_evaluator(pattern)

        # Convert to float if tensor
        if isinstance(energy, torch.Tensor):
            energy = energy.item()

        if energy < best_energy:
            best_energy = energy
            best_pattern = pattern

    if best_pattern is None:
        raise ValueError("No valid pattern found during energy-aware selection.")

    # Step 3: Return energy-optimal result
    result = SearchResult(
        best_pattern=best_pattern,
        neighbor_indices=basic_result.neighbor_indices,
        neighbor_distances=basic_result.neighbor_distances,
        selection_score=best_energy,  # Energy as score (lower = better)
    )

    return result


def search_with_entropy_selection(
    current_state: torch.Tensor,
    memory_patterns: torch.Tensor,
    entropy_evaluator: Callable[[torch.Tensor], float],
    k: int = 5,
    metric: str = "cosine",
) -> SearchResult:
    """
    Enhanced k-NN search with entropy-based selection.

    Instead of just taking nearest neighbor, evaluates each candidate
    by estimated entropy reduction.

    Args:
        current_state: Current embedding
        memory_patterns: Stored patterns
        entropy_evaluator: function that estimates entropy if pattern used as goal
                          Signature: (pattern: Tensor) -> float
        k: Number of neighbors to retrieve
        metric: Distance metric

    Returns:
        SearchResult with pattern that maximizes entropy reduction
    """
    # Get k nearest neighbors
    basic_result = knn_search(current_state, memory_patterns, k, metric)

    # Evaluate entropy for each neighbor
    best_pattern = None
    best_entropy = float("inf")

    for idx in basic_result.neighbor_indices:
        pattern = memory_patterns[idx]
        estimated_entropy = entropy_evaluator(pattern)

        if estimated_entropy < best_entropy:
            best_entropy = estimated_entropy
            best_pattern = pattern

    if best_pattern is None:
        raise ValueError("No valid pattern found during entropy-based selection.")

    # Update result with entropy-based selection
    result = SearchResult(
        best_pattern=best_pattern,
        neighbor_indices=basic_result.neighbor_indices,
        neighbor_distances=basic_result.neighbor_distances,
        selection_score=best_entropy,  # Use entropy as score
    )

    return result


def multi_hop_search(
    current_state: torch.Tensor,
    memory_patterns: torch.Tensor,
    num_hops: int = 2,
    k: int = 5,
    metric: str = "cosine",
) -> List[SearchResult]:
    """
    Multi-hop search: iteratively retrieve neighbors of neighbors.

    This can help escape local basins by exploring more distant regions
    of the manifold.

    Args:
        current_state: Starting state
        memory_patterns: Stored patterns
        num_hops: Number of search hops
        k: Neighbors per hop
        metric: Distance metric

    Returns:
        List of SearchResults, one per hop
    """
    results = []
    state = current_state

    for _ in range(num_hops):
        result = knn_search(state, memory_patterns, k, metric)
        results.append(result)

        # Move to best neighbor for next hop
        state = result.best_pattern.unsqueeze(0)

    return results
