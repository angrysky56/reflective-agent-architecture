import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path so we can import src
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.embeddings.embedding_factory import EmbeddingFactory  # noqa: E402
from src.vectordb_migrate.integrations.chroma_migrator import ChromaMigrator  # noqa: E402
from src.vectordb_migrate.migration import MigrationDetector  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("force-migration")


def force_migration() -> None:
    print("=== FORCE MIGRATION SCRIPT ===")

    # 0. Load Configuration
    load_dotenv()
    provider_name = os.getenv("EMBEDDING_PROVIDER", "openrouter")
    model_name = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")

    print("Loading config from environment...")
    print(f"Provider: {provider_name}")
    print(f"Model: {model_name}")

    try:
        # Create embedding provider to get current dimension
        embedding_model = EmbeddingFactory.create(
            provider_name=provider_name, model_name=model_name
        )
        expected_dim = embedding_model.get_sentence_embedding_dimension()
        print(f"Detected Target Dimension: {expected_dim}")
    except Exception as e:
        print(f"ERROR: Failed to initialize embedding model: {e}")
        return

    # 1. Setup Paths
    project_root = Path(".").resolve()
    projections_dir = project_root / "src" / "embeddings" / "projections"
    chroma_path = project_root / "chroma_data"

    print(f"Project Root: {project_root}")
    print(f"Projections Dir: {projections_dir}")
    print(f"Chroma Path: {chroma_path}")

    if not chroma_path.exists():
        print(f"ERROR: Chroma path {chroma_path} does not exist!")
        return

    # 2. Init Detector
    detector = MigrationDetector(projections_dir)
    print(f"Registry loaded ({len(detector.registry)} entries)")

    # 3. Init Migrator
    migrator = ChromaMigrator(str(chroma_path))

    # 4. Check Mismatches
    mismatches = migrator.check_dimensions(expected_dim)
    print(f"Found {len(mismatches)} mismatches against {expected_dim}D")

    if not mismatches:
        print("✓ Database is already up to date!")
        return

    # 5. Execute Migration
    for m in mismatches:
        collection_name = m["name"]
        current_dim = m["current_dim"]
        print(f"\n>> Migrating {collection_name} ({current_dim} -> {expected_dim})...")

        proj_path = detector.find_projection(current_dim, expected_dim)
        if not proj_path:
            print(
                f"SKIPPING {collection_name}: No projection found for {current_dim} -> {expected_dim}"
            )
            continue

        print(f"Using projection: {proj_path.name}")
        try:
            result = migrator.migrate_collection(
                collection_name=collection_name,
                projection_path=proj_path,
                batch_size=100,
                auto_backup=True,
                verify=True,
            )
            print(f"STATUS: {result['status']}")
            print(f"VECTORS MIGRATED: {result['n_migrated']}")
            print(f"VERIFICATION MSE: {result.get('verification', {}).get('mse')}")
        except Exception as e:
            print(f"FAILED to migrate {collection_name}: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    force_migration()
