"""
Advanced loss functions for embedding migration.

Supports multiple objectives:
- MSE: Reconstruction accuracy
- Cosine Similarity: Angular relationship preservation
- Triplet: Maintain relative distances
- Hybrid: Weighted combination
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as f


class CosineSimilarityLoss(nn.Module):
    """
    Cosine similarity loss for preserving angular relationships.

    This is particularly important for vector databases that use
    cosine similarity for semantic search.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute negative cosine similarity (lower = more similar).

        Args:
            predicted: Transformed embeddings (N, target_dim)
            target: Ground truth embeddings (N, target_dim)

        Returns:
            Loss value
        """
        # Normalize
        predicted_norm = f.normalize(predicted, p=2, dim=1)
        target_norm = f.normalize(target, p=2, dim=1)

        # Cosine similarity
        similarity = (predicted_norm * target_norm).sum(dim=1)

        # Return negative (we want to maximize similarity, minimize loss)
        return 1 - similarity.mean()


class TripletPreservationLoss(nn.Module):
    """
    Preserve relative distances between embeddings.

    For each triplet (anchor, positive, negative):
    - Positive should be closer than negative
    - Relative ordering should be preserved
    """

    def __init__(self, margin: float = 0.1) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        anchor_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute triplet loss for relative distance preservation.

        Args:
            predicted: Transformed embeddings
            target: Ground truth embeddings
            anchor_idx: Optional anchor indices for controlled triplets

        Returns:
            Loss value
        """
        if anchor_idx is None:
            # Random triplets
            n = predicted.size(0)
            anchor_idx = torch.randperm(n)[: n // 3]

        # Compute pairwise distances (both spaces)
        pred_dist = torch.cdist(predicted, predicted)
        target_dist = torch.cdist(target, target)

        # For each anchor, find closest and furthest
        for idx in anchor_idx:
            # Sort by target distance
            sorted_idx = torch.argsort(target_dist[idx])
            positive_idx = sorted_idx[1]  # Closest (skip self at 0)
            negative_idx = sorted_idx[-1]  # Furthest

            # Check if ordering is preserved in predicted space
            pred_pos_dist = pred_dist[idx, positive_idx]
            pred_neg_dist = pred_dist[idx, negative_idx]

        # Triplet loss: ensure positive closer than negative
        loss = f.relu(pred_pos_dist - pred_neg_dist + self.margin)

        return loss.mean()


class HybridLoss(nn.Module):
    """
    Hybrid loss combining multiple objectives.

    Default configuration:
    - 70% MSE (reconstruction)
    - 20% Cosine similarity (angular preservation)
    - 10% Triplet (relative distance preservation)
    """

    def __init__(
        self,
        mse_weight: float = 0.7,
        cosine_weight: float = 0.2,
        triplet_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.triplet_weight = triplet_weight

        self.mse = nn.MSELoss()
        self.cosine = CosineSimilarityLoss()
        self.triplet = TripletPreservationLoss()

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted combination of losses.

        Args:
            predicted: Transformed embeddings
            target: Ground truth embeddings

        Returns:
            Combined loss value
        """
        loss = 0.0

        if self.mse_weight > 0:
            loss += self.mse_weight * self.mse(predicted, target)

        if self.cosine_weight > 0:
            loss += self.cosine_weight * self.cosine(predicted, target)

        if self.triplet_weight > 0 and predicted.size(0) >= 3:
            loss += self.triplet_weight * self.triplet(predicted, target)

        return loss


def get_loss_function(
    loss_type: str = "mse",
    **kwargs: Any,
) -> nn.Module:
    """
    Factory function for loss functions.

    Args:
        loss_type: Type of loss ('mse', 'cosine', 'hybrid', 'triplet')
        **kwargs: Additional parameters for the loss function

    Returns:
        Loss function module
    """
    loss_functions = {
        "mse": nn.MSELoss,
        "cosine": CosineSimilarityLoss,
        "hybrid": HybridLoss,
        "triplet": TripletPreservationLoss,
    }

    if loss_type not in loss_functions:
        raise ValueError(
            f"Unknown loss type: {loss_type}. " f"Available: {list(loss_functions.keys())}"
        )

    return loss_functions[loss_type](**kwargs)
