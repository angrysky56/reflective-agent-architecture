
# VectorDB Migration System

This package provides a robust system for migrating vector embeddings between different models (e.g., BGE -> OpenAI) while preserving semantic relationships using linear projection.

## Features

- **Auto-Detection:** Automatically detects dimension mismatches on server startup.
- **Auto-Migration:** Seamlessly migrates collections if a compatible projection exists.
- **Learned Projections:** Uses linear layers to map embeddings from Dimension A to Dimension B.
- **Safety:** Auto-backups ChromaDB before any modification.

## How it Works

1. **Server Startup:** The `CWDRAABridge` initializes the server context.
2. **Dimension Check:** `_check_and_migrate_embeddings` compares the current embedding model dimension with the stored collection metadata.
3. **Migration:**
    - If a mismatch is found (e.g., 1024 vs 1536), it looks for a pre-trained projection in `src/embeddings/projections/registry.json`.
    - If found, it loads the projection, transforms all vectors, and updates the collection.
    - If not found, it can auto-train a new projection using cached text samples (if enabled).

## Manual Migration (Fallback)

If the auto-migration fails or you need to perform a migration offline, use the included utility script:

```bash
python src/scripts/force_migration.py
```

This script will:
1. Scan your local `chroma_data` for dimension mismatches.
2. Attempt to find a suitable projection.
3. Perform the migration with detailed logging.
4. Verify the quality of the migration (MSE/Cosine Similarity).

## Directory Structure

- `src/vectordb_migrate/`: Core migration package.
- `src/embeddings/projections/`: Registry and model weights (`.pt` files).
- `src/scripts/force_migration.py`: Manual fallback script.
