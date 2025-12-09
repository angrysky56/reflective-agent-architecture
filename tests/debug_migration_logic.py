import logging
from pathlib import Path

from src.vectordb_migrate.integrations.chroma_migrator import ChromaMigrator
from src.vectordb_migrate.migration import MigrationDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug-migration")


def test_migration_logic():
    print("=== Testing Migration Logic ===")

    # 1. Setup Paths
    project_root = Path(".").resolve()
    projections_dir = project_root / "src" / "embeddings" / "projections"
    chroma_path = project_root / "chroma_data"

    print(f"Project Root: {project_root}")
    print(f"Projections Dir: {projections_dir}")
    print(f"Chroma Path: {chroma_path}")

    # 2. Test MigrationDetector
    print("\n--- Testing MigrationDetector ---")
    detector = MigrationDetector(projections_dir)
    print(f"Registry loaded ({len(detector.registry)} entries)")

    # Check find_projection
    print("Finding projection for 1024 -> 1536...")
    proj_path = detector.find_projection(1024, 1536)
    if proj_path:
        print(f"SUCCESS: Found projection at {proj_path}")
    else:
        print("FAILURE: Projection not found")

    # 3. Test ChromaMigrator
    print(f"\n--- Testing ChromaMigrator at {chroma_path} ---")
    if not chroma_path.exists():
        print(f"ERROR: Chroma path {chroma_path} does not exist!")
        return

    migrator = ChromaMigrator(str(chroma_path))
    mismatches = migrator.check_dimensions(1536)
    print(f"Found {len(mismatches)} mismatches")
    for m in mismatches:
        print(f"  - {m['name']}: {m['current_dim']} -> 1536")


if __name__ == "__main__":
    test_migration_logic()
