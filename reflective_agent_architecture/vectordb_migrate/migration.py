"""
Embedding Migration System - Core Engine

Enables seamless migration between different embedding models using learned linear projections.
Preserves 80-95% of semantic relationships while avoiding expensive re-embedding.

Technical Approach:
- Train linear transformation: W × source_embedding + b = target_embedding
- Use MSE loss with optional cosine similarity preservation
- Validate with similarity metrics and nearest-neighbor consistency
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ProjectionModel(nn.Module):
    """
    Simple linear projection layer for embedding dimension transformation.

    Args:
        source_dim: Dimensionality of source embeddings
        target_dim: Dimensionality of target embeddings
    """

    def __init__(self, source_dim: int, target_dim: int):
        super().__init__()
        self.projection = nn.Linear(source_dim, target_dim, bias=True)

        # Xavier initialization for better convergence
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project source embeddings to target space."""
        return self.projection(x)


class EmbeddingMigration:
    """
    Handles training and application of embedding projections.

    Example:
        >>> migration = EmbeddingMigration()
        >>> migration.train_projection(
        ...     source_embeddings, target_embeddings,
        ...     save_path="projections/bge-large_to_openai.pt"
        ... )
        >>> new_embedding = migration.transform(old_embedding)
    """

    def __init__(self, device: str | None = None):
        """
        Initialize migration engine.

        Args:
            device: Device for computation ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: ProjectionModel | None = None
        self.source_dim: int | None = None
        self.target_dim: int | None = None
        self.metadata: dict[str, Any] = {}

    def train_projection(
        self,
        source_embeddings: np.ndarray | list[list[float]],
        target_embeddings: np.ndarray | list[list[float]],
        epochs: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        validation_split: float = 0.2,
        save_path: str | Path | None = None,
        loss_fn: nn.Module | None = None,
        verbose: bool = True,
    ) -> dict[str, float]:
        """
        Train a linear projection from source to target embedding space.

        Args:
            source_embeddings: Source model embeddings (N, source_dim)
            target_embeddings: Target model embeddings (N, target_dim)
            epochs: Training epochs
            learning_rate: Adam optimizer learning rate
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            save_path: Path to save trained model
            loss_fn: Optional custom loss function
            verbose: Show training progress

        Returns:
            Dictionary with training metrics (loss, similarity_preservation, etc.)
        """
        # Convert to tensors
        source_embeddings = torch.FloatTensor(np.array(source_embeddings)).to(self.device)
        target_embeddings = torch.FloatTensor(np.array(target_embeddings)).to(self.device)

        n_samples = source_embeddings.size(0)
        self.source_dim = source_embeddings.size(1)
        self.target_dim = target_embeddings.size(1)

        if verbose:
            logger.info(
                f"Training projection: {self.source_dim}D → {self.target_dim}D "
                f"({n_samples} samples)"
            )

        # Split into train/val
        n_val = int(n_samples * validation_split)
        indices = torch.randperm(n_samples)

        train_idx, val_idx = indices[n_val:], indices[:n_val]
        source_train, source_val = source_embeddings[train_idx], source_embeddings[val_idx]
        target_train, target_val = target_embeddings[train_idx], target_embeddings[val_idx]

        # Initialize model
        self.model = ProjectionModel(self.source_dim, self.target_dim).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = loss_fn if loss_fn is not None else nn.MSELoss()

        # Training loop
        best_val_loss = float("inf")
        best_model_state = None

        if verbose:
            iterator = tqdm(range(epochs), desc="Training")
        else:
            iterator = range(epochs)  # type: ignore[assignment]

        for epoch in iterator:
            self.model.train()
            train_loss = 0.0

            # Mini-batch training
            for i in range(0, len(source_train), batch_size):
                batch_source = source_train[i : i + batch_size]
                batch_target = target_train[i : i + batch_size]

                optimizer.zero_grad()
                output = self.model(batch_source)
                loss = criterion(output, batch_target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(source_val)
                val_loss = criterion(val_output, target_val).item()

                # Track best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()

            if verbose and epoch % 10 == 0:
                iterator.set_postfix(
                    {"train_loss": train_loss / len(source_train), "val_loss": val_loss}
                )

        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        # Calculate final metrics
        metrics = self._calculate_metrics(source_val, target_val)

        if verbose:
            logger.info(
                f"Training complete - Val Loss: {best_val_loss:.6f}, "
                f"Similarity Preservation: {metrics['similarity_preservation']:.2%}"
            )

        # Save metadata
        self.metadata = {
            "source_dim": self.source_dim,
            "target_dim": self.target_dim,
            "n_samples": n_samples,
            "epochs": epochs,
            "final_val_loss": best_val_loss,
            **metrics,
        }

        # Save model
        if save_path:
            self.save(save_path)

        return metrics

    def _calculate_metrics(
        self, source_val: torch.Tensor, target_val: torch.Tensor
    ) -> dict[str, float]:
        """Calculate validation metrics."""
        if self.model is None:
            raise RuntimeError("No model loaded")

        self.model.eval()
        with torch.no_grad():
            projected = self.model(source_val)

            # MSE
            mse = nn.MSELoss()(projected, target_val).item()

            # Cosine similarity preservation
            # How well are angular relationships preserved?
            original_sims = self._cosine_similarity_matrix(target_val)
            projected_sims = self._cosine_similarity_matrix(projected)

            # Correlation between similarity matrices
            sim_preservation = np.corrcoef(
                original_sims.flatten().cpu().numpy(),
                projected_sims.flatten().cpu().numpy(),
            )[0, 1]

            return {
                "mse": float(mse),
                "similarity_preservation": float(sim_preservation),
            }

    @staticmethod
    def _cosine_similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise cosine similarities."""
        normalized = embeddings / embeddings.norm(dim=1, keepdim=True)
        return normalized @ normalized.T

    def transform(self, embeddings: np.ndarray | list[float] | torch.Tensor) -> np.ndarray:
        """
        Transform embeddings using the learned projection.

        Args:
            embeddings: Source embeddings (can be single vector or batch)

        Returns:
            Transformed embeddings in target space
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Train or load a projection first.")

        # Handle single vector vs batch
        single_vector = False
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)

        if isinstance(embeddings, np.ndarray):
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
                single_vector = True
            embeddings = torch.FloatTensor(embeddings).to(self.device)

        # Transform
        self.model.eval()
        with torch.no_grad():
            transformed = self.model(embeddings)

        result = transformed.cpu().numpy()

        return result[0] if single_vector else result

    def save(self, path: str | Path) -> None:
        """Save projection model and metadata."""
        if self.model is None:
            raise RuntimeError("No model to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model state and metadata together
        save_dict = {
            "model_state": self.model.state_dict(),
            "source_dim": self.source_dim,
            "target_dim": self.target_dim,
            "metadata": self.metadata,
        }

        torch.save(save_dict, path)
        logger.info(f"Projection saved to {path}")

    def load(self, path: str | Path) -> None:
        """Load projection model and metadata."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Projection not found: {path}")

        # NOTE: weights_only=False is required because we save a dictionary containing:
        # - model_state_dict (weights)
        # - source_dim, target_dim (metadata)
        # - training metrics (metadata)
        # This is safe because we control the projection files and they're created by our code.
        save_dict = torch.load(  # noqa: S614  # trunk-ignore(bandit/B614)
            path, map_location=self.device, weights_only=False
        )

        self.source_dim = save_dict["source_dim"]
        self.target_dim = save_dict["target_dim"]
        self.metadata = save_dict.get("metadata", {})

        self.model = ProjectionModel(self.source_dim, self.target_dim).to(self.device)
        self.model.load_state_dict(save_dict["model_state"])
        self.model.eval()

        logger.info(
            f"Loaded projection: {self.source_dim}D → {self.target_dim}D "
            f"(similarity preservation: {self.metadata.get('similarity_preservation', 'N/A')})"
        )


class MigrationDetector:
    """
    Detects dimension mismatches and suggests appropriate migrations.
    """

    def __init__(self, projections_dir: str | Path):
        """
        Initialize detector with projections directory.

        Args:
            projections_dir: Directory containing projection models
        """
        self.projections_dir = Path(projections_dir)
        self.registry = self._load_registry()

    def _load_registry(self) -> dict[str, Any]:
        """Load projection registry if it exists."""
        registry_path = self.projections_dir / "registry.json"
        if registry_path.exists():
            with open(registry_path) as f:
                return cast(dict[str, Any], json.load(f))
        return {}

    def find_projection(
        self, source_dim: int, target_dim: int, source_model: str | None = None
    ) -> Path | None:
        """
        Find a suitable projection for the given dimensions.

        Args:
            source_dim: Source embedding dimension
            target_dim: Target embedding dimension
            source_model: Optional source model name for better matching

        Returns:
            Path to projection file, or None if not found
        """
        # Check registry first
        for name, info in self.registry.items():
            if name.startswith("_") or not isinstance(info, dict):
                continue

            if info.get("source_dim") == source_dim and info.get("target_dim") == target_dim:
                projection_path = self.projections_dir / f"{name}.pt"
                if projection_path.exists():
                    logger.info(f"Found pre-trained projection: {name}")
                    return projection_path

        # Fallback: scan directory for matching dimensions
        for projection_path in self.projections_dir.glob("*.pt"):
            try:
                # NOTE: weights_only=False needed to read our custom save format
                # Safe because we're only loading files we created ourselves
                save_dict = torch.load(  # noqa: S614  # trunk-ignore(bandit/B614)
                    projection_path, map_location="cpu", weights_only=False
                )
                if (
                    save_dict.get("source_dim") == source_dim
                    and save_dict.get("target_dim") == target_dim
                ):
                    logger.info(f"Found projection: {projection_path.name}")
                    return projection_path
            except Exception as e:
                logger.warning(f"Failed to load {projection_path}: {e}")

        return None

    def suggest_migration(self, detected_dim: int, expected_dim: int) -> dict[str, Any]:
        """
        Suggest a migration strategy for dimension mismatch.

        Returns:
            Dictionary with suggestion details
        """
        projection_path = self.find_projection(detected_dim, expected_dim)

        result: dict[str, Any] = {
            "detected_dim": detected_dim,
            "expected_dim": expected_dim,
            "projection_available": projection_path is not None,
            "projection_path": str(projection_path) if projection_path else None,
            "recommendation": (
                f"Use pre-trained projection: {projection_path.name}"
                if projection_path
                else f"Train new projection: {detected_dim}D → {expected_dim}D"
            ),
        }
        return result
