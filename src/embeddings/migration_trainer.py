"""
Embedding Migration Trainer

Utility for training new projection models between embedding spaces.
Can be used as a module or standalone script.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.embeddings.base_embedding_provider import BaseEmbeddingProvider
from src.embeddings.embedding_factory import EmbeddingFactory
from src.vectordb_migrate.migration import EmbeddingMigration

logger = logging.getLogger(__name__)


def generate_sample_texts(n_samples: int = 1000) -> list[str]:
    """
    Generate diverse sample texts for training projections.

    Uses a mix of:
    - Technical terms
    - Common concepts
    - Abstract ideas
    - Domain-specific vocabulary

    Args:
        n_samples: Number of sample texts to generate

    Returns:
        List of sample text strings
    """
    # Base vocabulary pools
    technical = [
        "algorithm",
        "database",
        "optimization",
        "neural network",
        "embedding",
        "vector space",
        "similarity",
        "clustering",
        "classification",
        "regression",
        "transformer",
        "attention mechanism",
        "gradient descent",
        "backpropagation",
        "inference",
        "training",
        "validation",
        "hyperparameter",
        "architecture",
    ]

    concepts = [
        "knowledge",
        "understanding",
        "reasoning",
        "logic",
        "creativity",
        "intelligence",
        "learning",
        "memory",
        "perception",
        "cognition",
        "pattern",
        "structure",
        "relationship",
        "hierarchy",
        "abstraction",
        "synthesis",
        "analysis",
        "evaluation",
        "comparison",
        "transformation",
    ]

    domains = [
        "scientific research",
        "software engineering",
        "data analysis",
        "machine learning",
        "artificial intelligence",
        "natural language processing",
        "computer vision",
        "robotics",
        "quantum computing",
        "bioinformatics",
        "cognitive science",
        "philosophy",
        "mathematics",
        "physics",
        "linguistics",
    ]

    actions = [
        "analyze",
        "evaluate",
        "compare",
        "synthesize",
        "decompose",
        "optimize",
        "transform",
        "visualize",
        "model",
        "predict",
        "detect",
        "classify",
        "cluster",
        "generate",
        "interpret",
    ]

    # Generate samples
    samples = []

    # 1. Single terms
    samples.extend(technical[: n_samples // 4])
    samples.extend(concepts[: n_samples // 4])

    # 2. Two-word combinations
    import random

    # Security: random is safe here as we're just generating sample text, not crypto keys
    for _ in range(n_samples // 4):
        samples.append(
            f"{random.choice(actions)} {random.choice(concepts)}"  # noqa: S311  # trunk-ignore(bandit/B311)
        )

    # 3. Phrases
    for _ in range(n_samples // 4):
        samples.append(
            f"{random.choice(technical)} for {random.choice(domains)}"  # noqa: S311  # trunk-ignore(bandit/B311)
        )

    # 4. Questions
    question_templates = [
        f"How does {random.choice(technical)} work?",  # noqa: S311  # trunk-ignore(bandit/B311)
        f"What is the relationship between {random.choice(concepts)} and {random.choice(concepts)}?",  # noqa: S311  # trunk-ignore(bandit/B311)
        f"Can we {random.choice(actions)} {random.choice(technical)}?",  # noqa: S311  # trunk-ignore(bandit/B311)
    ]
    samples.extend(question_templates * (n_samples // 12))

    return samples[:n_samples]


def train_projection(
    source_provider: str,
    source_model: str,
    target_provider: str,
    target_model: str,
    n_samples: int = 1000,
    output_dir: str | Path = "src/embeddings/projections",
    device: str | None = None,
) -> dict[str, Any]:
    """
    Train a projection between two embedding models.

    Args:
        source_provider: Source embedding provider name
        source_model: Source model name
        target_provider: Target embedding provider name
        target_model: Target model name
        n_samples: Number of training samples
        output_dir: Directory to save projection
        device: Device for training (cuda/cpu)

    Returns:
        Training results and metrics
    """
    logger.info(f"Training projection: {source_model} → {target_model}")

    # Initialize embedding providers
    logger.info("Loading source model...")
    source_embedding: BaseEmbeddingProvider = EmbeddingFactory.create(
        provider_name=source_provider, model_name=source_model
    )

    logger.info("Loading target model...")
    target_embedding: BaseEmbeddingProvider = EmbeddingFactory.create(
        provider_name=target_provider, model_name=target_model
    )

    # Generate sample texts
    logger.info(f"Generating {n_samples} sample texts...")
    sample_texts = generate_sample_texts(n_samples)

    # Generate embeddings
    logger.info("Generating source embeddings...")
    source_embeddings = [source_embedding.encode(text) for text in sample_texts]

    logger.info("Generating target embeddings...")
    target_embeddings = [target_embedding.encode(text) for text in sample_texts]

    # Convert to numpy arrays
    source_embeddings = np.array(source_embeddings)
    target_embeddings = np.array(target_embeddings)

    logger.info(
        f"Embedding dimensions: {source_embeddings.shape[1]} → {target_embeddings.shape[1]}"
    )

    # Train projection
    migration = EmbeddingMigration(device=device)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    source_name = source_model.replace("/", "_").replace(":", "_")
    target_name = target_model.replace("/", "_").replace(":", "_")
    projection_name = f"{source_name}_to_{target_name}"
    projection_path = output_dir / f"{projection_name}.pt"

    # Train
    from src.vectordb_migrate.loss_functions import HybridLoss

    metrics = migration.train_projection(
        source_embeddings=source_embeddings,
        target_embeddings=target_embeddings,
        epochs=100,
        save_path=projection_path,
        loss_fn=HybridLoss(
            mse_weight=0.7, cosine_weight=0.3, triplet_weight=0.0
        ),  # 70% MSE, 30% Cosine
        verbose=True,
    )

    # Update registry
    registry_path = output_dir / "registry.json"
    registry = {}
    if registry_path.exists():
        with open(registry_path) as f:
            registry = json.load(f)

    registry[projection_name] = {
        "source_provider": source_provider,
        "source_model": source_model,
        "source_dim": int(source_embeddings.shape[1]),
        "target_provider": target_provider,
        "target_model": target_model,
        "target_dim": int(target_embeddings.shape[1]),
        "n_samples": n_samples,
        "similarity_preservation": float(metrics["similarity_preservation"]),
        "mse": float(metrics["mse"]),
        "file": f"{projection_name}.pt",
    }

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    logger.info(f"Projection saved: {projection_path}")
    logger.info(f"Registry updated: {registry_path}")

    return {
        "projection_path": str(projection_path),
        "metrics": metrics,
    }


def main() -> None:
    """CLI entry point for training projections."""
    parser = argparse.ArgumentParser(description="Train embedding projection for model migration")

    parser.add_argument(
        "--source-provider",
        type=str,
        default="sentence-transformers",
        help="Source embedding provider",
    )
    parser.add_argument(
        "--source-model",
        type=str,
        required=True,
        help="Source model name",
    )
    parser.add_argument(
        "--target-provider",
        type=str,
        default="sentence-transformers",
        help="Target embedding provider",
    )
    parser.add_argument(
        "--target-model",
        type=str,
        required=True,
        help="Target model name",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of training samples",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/embeddings/projections",
        help="Output directory for projection",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="Device for training (default: auto)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Train
    result = train_projection(
        source_provider=args.source_provider,
        source_model=args.source_model,
        target_provider=args.target_provider,
        target_model=args.target_model,
        n_samples=args.samples,
        output_dir=args.output_dir,
        device=args.device,
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Projection: {result['projection_path']}")
    print(f"MSE: {result['metrics']['mse']:.6f}")
    print(f"Similarity Preservation: {result['metrics']['similarity_preservation']:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
