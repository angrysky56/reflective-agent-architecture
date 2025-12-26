"""
Chroma Vector Database Migrator

Applies embedding migrations to Chroma collections with automatic backup and rollback support.
Handles batch processing for large collections efficiently.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from tqdm import tqdm

from ..migration import EmbeddingMigration

logger = logging.getLogger(__name__)


class ChromaMigrator:
    """
    Handles migration of Chroma vector collections to new embedding dimensions.

    Features:
    - Automatic backup before migration
    - Batch processing for efficiency
    - Progress reporting
    - Rollback capability
    - Validation checks

    Example:
        >>> migrator = ChromaMigrator("/path/to/chroma_data")
        >>> migrator.migrate_collection(
        ...     collection_name="thought_nodes",
        ...     projection_path="projections/bge-large_to_openai.pt"
        ... )
    """

    def __init__(self, chroma_path: str | Path):
        """
        Initialize migrator for a Chroma database.

        Args:
            chroma_path: Path to Chroma persistent storage
        """
        self.chroma_path = Path(chroma_path)
        if not self.chroma_path.exists():
            raise FileNotFoundError(f"Chroma database not found: {chroma_path}")

        self.client = chromadb.PersistentClient(path=str(self.chroma_path))

    def migrate_collection(
        self,
        collection_name: str,
        projection_path: str | Path,
        batch_size: int = 100,
        auto_backup: bool = True,
        verify: bool = True,
    ) -> dict[str, Any]:
        """
        Migrate a collection to new embedding dimensions using a trained projection.

        Args:
            collection_name: Name of collection to migrate
            projection_path: Path to trained projection model
            batch_size: Number of vectors to process at once
            auto_backup: Automatically backup before migration
            verify: Verify migration quality after completion

        Returns:
            Migration statistics and results
        """
        logger.info(f"Starting migration for collection: {collection_name}")

        # Backup
        backup_path = None
        if auto_backup:
            backup_path = self._backup_database()
            logger.info(f"Backup created at: {backup_path}")

        try:
            # Load projection
            migration = EmbeddingMigration()
            migration.load(projection_path)

            source_dim = migration.source_dim
            target_dim = migration.target_dim

            logger.info(f"Loaded projection: {source_dim}D → {target_dim}D")

            # Get collection
            collection = self.client.get_collection(collection_name)

            # Get all embeddings
            results = collection.get(include=["embeddings", "metadatas", "documents"])

            if len(results["embeddings"]) == 0:
                logger.warning(f"Collection {collection_name} is empty, nothing to migrate")
                return {"status": "empty", "n_migrated": 0}

            embeddings = np.array(results["embeddings"])
            ids = results["ids"]
            metadatas = results["metadatas"]
            documents = results["documents"]

            n_total = len(embeddings)
            current_dim = embeddings.shape[1]

            logger.info(f"Found {n_total} vectors of dimension {current_dim}")

            # Validate dimensions
            if current_dim != source_dim:
                raise ValueError(
                    f"Dimension mismatch: Collection has {current_dim}D, "
                    f"projection expects {source_dim}D"
                )

            # Delete and recreate collection with new dimension
            self.client.delete_collection(collection_name)
            new_collection = self.client.create_collection(
                name=collection_name,
                metadata=collection.metadata,
            )

            # Migrate in batches
            logger.info(f"Migrating {n_total} vectors in batches of {batch_size}...")

            for i in tqdm(range(0, n_total, batch_size), desc="Migrating"):
                batch_end = min(i + batch_size, n_total)

                batch_embeddings = embeddings[i:batch_end]
                batch_ids = ids[i:batch_end]
                batch_metadatas = metadatas[i:batch_end] if metadatas else None
                batch_documents = documents[i:batch_end] if documents else None

                # Transform embeddings
                transformed = migration.transform(batch_embeddings)

                # Add to new collection
                new_collection.add(
                    ids=batch_ids,
                    embeddings=transformed.tolist(),
                    metadatas=batch_metadatas,
                    documents=batch_documents,
                )

            logger.info(f"Migration complete: {n_total} vectors migrated")

            # Verification
            verification_results = {}
            if verify:
                verification_results = self._verify_migration(
                    collection_name, embeddings[:100], migration
                )

            return {
                "status": "success",
                "n_migrated": n_total,
                "source_dim": source_dim,
                "target_dim": target_dim,
                "backup_path": str(backup_path) if backup_path else None,
                "verification": verification_results,
            }

        except Exception as e:
            logger.error(f"Migration failed: {e}")

            # Rollback if we have a backup
            if backup_path:
                logger.info("Attempting rollback from backup...")
                self._rollback_from_backup(backup_path)

            raise

    def _backup_database(self) -> Path:
        """Create a timestamped backup of the entire Chroma database."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.chroma_path.parent / f"chroma_backup_{timestamp}"

        logger.info(f"Creating backup: {backup_dir}")
        shutil.copytree(self.chroma_path, backup_dir)

        return backup_dir

    def _rollback_from_backup(self, backup_path: Path) -> None:
        """Restore database from backup."""
        logger.warning(f"Rolling back from backup: {backup_path}")

        # Remove current database
        shutil.rmtree(self.chroma_path)

        # Restore from backup
        shutil.copytree(backup_path, self.chroma_path)

        logger.info("Rollback complete")

    def _verify_migration(
        self,
        collection_name: str,
        original_embeddings: np.ndarray,
        migration: EmbeddingMigration,
        n_samples: int = 100,
    ) -> dict[str, Any]:
        """
        Verify migration quality by checking semantic similarity preservation.

        Args:
            collection_name: Name of migrated collection
            original_embeddings: Original embeddings (pre-migration)
            migration: Migration object with transform capability
            n_samples: Number of samples to check

        Returns:
            Verification metrics
        """
        logger.info("Verifying migration quality...")

        # Get migrated embeddings
        collection = self.client.get_collection(collection_name)
        results = collection.get(limit=n_samples, include=["embeddings"])
        migrated_embeddings = np.array(results["embeddings"])

        # Transform original embeddings
        expected_embeddings = migration.transform(original_embeddings[:n_samples])

        # Calculate accuracy
        mse = np.mean((migrated_embeddings - expected_embeddings) ** 2)

        # Similarity preservation check
        original_sims = self._cosine_similarity_matrix(original_embeddings[:n_samples])
        migrated_sims = self._cosine_similarity_matrix(migrated_embeddings)

        correlation = np.corrcoef(original_sims.flatten(), migrated_sims.flatten())[0, 1]

        logger.info(f"Verification - MSE: {mse:.6f}, Similarity Correlation: {correlation:.4f}")

        return {
            "mse": float(mse),
            "similarity_correlation": float(correlation),
            "n_verified": n_samples,
        }

    @staticmethod
    def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarities."""
        normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return normalized @ normalized.T

    def list_collections(self) -> list[dict[str, Any]]:
        """List all collections with their metadata."""
        collections = self.client.list_collections()

        result = []
        for collection in collections:
            count = collection.count()

            # Get sample embedding to determine dimension
            if count > 0:
                sample = collection.get(limit=1, include=["embeddings"])
                dim = len(sample["embeddings"][0]) if len(sample["embeddings"]) > 0 else None
            else:
                dim = None

            result.append(
                {
                    "name": collection.name,
                    "count": count,
                    "dimension": dim,
                    "metadata": collection.metadata,
                }
            )

        return result

    def check_dimensions(self, expected_dim: int) -> list[dict[str, Any]]:
        """
        Check all collections for dimension mismatches.

        Args:
            expected_dim: Expected embedding dimension

        Returns:
            List of collections with mismatches
        """
        collections = self.list_collections()
        mismatches = []

        for col in collections:
            if col["dimension"] and col["dimension"] != expected_dim:
                mismatches.append(
                    {
                        "name": col["name"],
                        "current_dim": col["dimension"],
                        "expected_dim": expected_dim,
                        "count": col["count"],
                    }
                )

        return mismatches
