"""
VectorDB-Migrate: Universal embedding migration for vector databases.

Zero-downtime model transitions using learned linear projections.
"""

__version__ = "0.1.0"
__author__ = "Tyler B. Hall, Claude Sonnet 4.5"

from .loss_functions import CosineSimilarityLoss, HybridLoss, get_loss_function
from .migration import EmbeddingMigration, MigrationDetector, ProjectionModel

__all__ = [
    "EmbeddingMigration",
    "MigrationDetector",
    "ProjectionModel",
    "HybridLoss",
    "CosineSimilarityLoss",
    "get_loss_function",
]

# Conditional import for Chroma integration
try:
    from .integrations.chroma_migrator import ChromaMigrator  # noqa: F401

    __all__.append("ChromaMigrator")
except ImportError as e:
    # ChromaDB not installed - skip integration
    print(f"DEBUG: Failed to import ChromaMigrator: {e}")
    pass
