"""
Embedding Forensics - Identify Unknown Embedding Models

Scans Chroma collections to identify which embedding models created which vectors,
even when metadata is missing. Uses dimension analysis, statistical fingerprinting,
and user confirmation to resolve mixed-model collections.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from reflective_agent_architecture.embeddings.embedding_factory import EmbeddingFactory

logger = logging.getLogger(__name__)


# Known model dimensions (expandable)
KNOWN_MODELS = {
    # Sentence Transformers
    384: [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
    ],
    768: [
        "sentence-transformers/all-mpnet-base-v2",
        "BAAI/bge-base-en-v1.5",
        "sentence-transformers/all-roberta-large-v1",
    ],
    1024: [
        "BAAI/bge-large-en-v1.5",
        "sentence-transformers/gtr-t5-large",
        "cohere/embed-english-v3.0",  # Note: dimension overlap with BGE
    ],
    # OpenAI
    1536: ["openai/text-embedding-ada-002", "openai/text-embedding-3-small"],
    3072: ["openai/text-embedding-3-large"],
    # Add more as needed
}


class EmbeddingForensics:
    """
    Analyze Chroma collections to identify unknown embedding models.

    Features:
    - Dimension-based grouping
    - Statistical fingerprinting
    - Interactive model confirmation
    - Batch migration planning
    """

    def __init__(self, chroma_path: str | Path):
        """
        Initialize forensics analyzer.

        Args:
            chroma_path: Path to Chroma database
        """
        self.chroma_path = Path(chroma_path)
        self.client = chromadb.PersistentClient(path=str(self.chroma_path))
        self._register_active_model()

    def _register_active_model(self) -> None:
        """Register currently active model from environment as a known model."""
        load_dotenv()
        provider_name = os.getenv("EMBEDDING_PROVIDER")
        model_name = os.getenv("EMBEDDING_MODEL")

        if not provider_name or not model_name:
            return

        try:
            embedding_model = EmbeddingFactory.create(
                provider_name=provider_name, model_name=model_name
            )
            dim = embedding_model.get_sentence_embedding_dimension()

            # Update global registry
            if dim not in KNOWN_MODELS:
                KNOWN_MODELS[dim] = []

            if model_name not in KNOWN_MODELS[dim]:
                KNOWN_MODELS[dim].append(model_name)
                logger.info(f"Dynamically registered active model: {model_name} ({dim}d)")

        except Exception as e:
            logger.warning(f"Failed to register active model for forensics: {e}")

    def scan_collection(self, collection_name: str, sample_size: int = 100) -> dict[str, Any]:
        """
        Scan a collection to identify embedding models used.

        Args:
            collection_name: Name of collection to scan
            sample_size: Number of vectors to sample for analysis

        Returns:
            Analysis results with detected models and statistics
        """
        logger.info(f"🔍 Scanning collection: {collection_name}")

        collection = self.client.get_collection(collection_name)

        # Get all embeddings (or sample if too large)
        total_count = collection.count()

        if total_count == 0:
            return {"status": "empty", "count": 0}

        logger.info(f"Total vectors: {total_count}")

        # Sample for analysis
        results = collection.get(
            limit=min(sample_size, total_count), include=["embeddings", "metadatas"]
        )

        embeddings = np.array(results["embeddings"])
        metadatas = results["metadatas"]

        # Group by dimension
        dimension_groups = self._group_by_dimension(embeddings, metadatas)

        # Analyze each dimension group
        analyses = {}
        for dim, group_data in dimension_groups.items():
            analyses[dim] = self._analyze_dimension_group(dim, group_data)

        return {
            "status": "analyzed",
            "collection": collection_name,
            "total_count": total_count,
            "sampled": len(embeddings),
            "dimension_groups": analyses,
        }

    def _group_by_dimension(
        self, embeddings: np.ndarray, metadatas: list[dict[str, Any]] | None
    ) -> dict[int, dict[str, Any]]:
        """Group embeddings by dimension."""
        groups: dict[int, dict[str, Any]] = defaultdict(lambda: {"indices": [], "metadatas": []})

        for i, emb in enumerate(embeddings):
            dim = len(emb)
            groups[dim]["indices"].append(i)
            if metadatas:
                groups[dim]["metadatas"].append(metadatas[i])

        return dict(groups)

    def _analyze_dimension_group(
        self, dimension: int, group_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Analyze a group of same-dimension embeddings.

        Returns statistical properties and likely models.
        """
        indices = group_data["indices"]
        metadatas = group_data["metadatas"]

        # Check if metadata has model info
        models_from_metadata = set()
        for meta in metadatas:
            if meta and "embedding_model" in meta:
                models_from_metadata.add(meta["embedding_model"])

        # Get candidate models for this dimension
        candidate_models = KNOWN_MODELS.get(dimension, [])

        return {
            "dimension": dimension,
            "count": len(indices),
            "models_from_metadata": list(models_from_metadata),
            "candidate_models": candidate_models,
            "confidence": "high" if models_from_metadata else "low",
        }

    def generate_migration_plan(
        self, scan_results: dict[str, Any], target_dim: int
    ) -> dict[str, Any]:
        """
        Generate a migration plan for mixed-model collections.

        Args:
            scan_results: Results from scan_collection()
            target_dim: Target embedding dimension

        Returns:
            Migration plan with steps for each dimension group
        """
        if scan_results["status"] == "empty":
            return {"status": "no_migration_needed", "reason": "Collection is empty"}

        dimension_groups = scan_results["dimension_groups"]

        # Check if already all target dimension
        if len(dimension_groups) == 1 and target_dim in dimension_groups:
            return {"status": "no_migration_needed", "reason": "Already at target dimension"}

        # Build migration steps
        steps = []
        for dim, analysis in dimension_groups.items():
            if dim == target_dim:
                continue  # Skip - already correct

            # Determine source model
            if analysis["models_from_metadata"]:
                source_model = analysis["models_from_metadata"][0]
                confidence = "high"
            elif analysis["candidate_models"]:
                source_model = analysis["candidate_models"][0]
                confidence = "low"
            else:
                source_model = None
                confidence = "none"

            steps.append(
                {
                    "source_dim": dim,
                    "target_dim": target_dim,
                    "count": analysis["count"],
                    "detected_model": source_model,
                    "confidence": confidence,
                    "action": "migrate" if source_model else "manual_identification_required",
                }
            )

        return {
            "status": "migration_planned",
            "collection": scan_results["collection"],
            "total_vectors": scan_results["total_count"],
            "steps": steps,
        }

    def interactive_resolve(self, scan_results: dict[str, Any]) -> dict[int, str]:
        """
        Interactively resolve unknown models with user input.

        Args:
            scan_results: Results from scan_collection()

        Returns:
            Map of dimension -> confirmed model name
        """
        dimension_groups = scan_results["dimension_groups"]
        resolved = {}

        print("\n" + "=" * 60)
        print("🔍 EMBEDDING FORENSICS - Model Identification")
        print("=" * 60)

        for dim, analysis in dimension_groups.items():
            print(f"\n📊 Dimension: {dim}D ({analysis['count']} vectors)")

            if analysis["models_from_metadata"]:
                print(f"✓ Detected from metadata: {analysis['models_from_metadata'][0]}")
                resolved[dim] = analysis["models_from_metadata"][0]
                continue

            if analysis["candidate_models"]:
                print("Possible models:")
                for i, model in enumerate(analysis["candidate_models"], 1):
                    print(f"  {i}. {model}")

                while True:
                    choice = input(
                        f"Select model (1-{len(analysis['candidate_models'])}) or type custom: "
                    )

                    if choice.isdigit() and 1 <= int(choice) <= len(analysis["candidate_models"]):
                        resolved[dim] = analysis["candidate_models"][int(choice) - 1]
                        break
                    elif choice.strip():
                        resolved[dim] = choice.strip()
                        break
                    else:
                        print("Invalid choice, please try again.")
            else:
                print("⚠ No known models for this dimension")
                model = input("Enter model name manually: ")
                resolved[dim] = model.strip()

        print("\n" + "=" * 60)
        print("✓ Model identification complete!")
        print("=" * 60)

        return resolved

    def save_forensics_report(
        self,
        scan_results: dict[str, Any],
        output_path: str | Path = "embedding_forensics_report.json",
    ) -> None:
        """Save forensics analysis to JSON file."""
        output_path = Path(output_path)

        with open(output_path, "w") as f:
            json.dump(scan_results, f, indent=2)

        logger.info(f"Forensics report saved: {output_path}")


def scan_all_collections(chroma_path: str | Path) -> dict[str, Any]:
    """
    Scan all collections in a Chroma database.

    Args:
        chroma_path: Path to Chroma database

    Returns:
        Analysis results for all collections
    """
    forensics = EmbeddingForensics(chroma_path)
    client = chromadb.PersistentClient(path=str(chroma_path))

    collections = client.list_collections()

    if not collections:
        logger.info("No collections found")
        return {"status": "no_collections"}

    results = {}
    for collection in tqdm(collections, desc="Scanning collections"):
        results[collection.name] = forensics.scan_collection(collection.name)

    return {
        "status": "complete",
        "chroma_path": str(chroma_path),
        "collections": results,
    }


def main() -> None:
    """CLI entry point for embedding forensics."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Scan Chroma database for unknown embedding models"
    )
    parser.add_argument("chroma_path", type=str, help="Path to Chroma database directory")
    parser.add_argument("--collection", type=str, help="Specific collection to scan (default: all)")
    parser.add_argument(
        "--interactive", action="store_true", help="Interactively resolve unknown models"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="embedding_forensics_report.json",
        help="Output file for report",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    forensics = EmbeddingForensics(args.chroma_path)

    if args.collection:
        # Scan specific collection
        results = forensics.scan_collection(args.collection)

        if args.interactive and results["status"] == "analyzed":
            resolved = forensics.interactive_resolve(results)
            results["resolved_models"] = resolved
    else:
        # Scan all collections
        results = scan_all_collections(args.chroma_path)

    # Save report
    forensics.save_forensics_report(results, args.output)

    print(f"\n✓ Analysis complete! Report saved to: {args.output}")


if __name__ == "__main__":
    main()
