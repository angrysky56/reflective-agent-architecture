# Embedding Migration System

Universal embedding migration for seamless model transitions without data loss.

## Overview

The Embedding Migration System allows you to change embedding models without re-processing your entire vector database. It uses **linear projection mapping** to transform embeddings between different dimensional spaces, preserving 80-95% of semantic relationships.

## ⚠️ Important Warnings

> [!WARNING]
> **Semantic Drift**: Migrations may result in 5-20% loss of semantic accuracy. The system preserves the majority of relationships but is not perfect.

> [!WARNING]
> **Automatic Backup**: The system automatically creates backups before migration, but always ensure you have additional backups of critical data.

> [!IMPORTANT]
> **Production Use**: Test migrations on a copy of your data before applying to production databases.

## How It Works

### Linear Projection Approach

1. **Training Phase**:
   - Sample texts are embedded with both old and new models
   - A simple linear transformation is trained: `W × old_embedding + b = new_embedding`
   - The model learns to map between embedding spaces

2. **Migration Phase**:
   - Each old embedding is transformed using the learned projection
   - Processing is highly efficient (~10,000 vectors/second)
   - Semantic relationships are largely preserved

3. **Quality Metrics**:
   - **MSE**: Reconstruction accuracy
   - **Similarity Preservation**: Maintained semantic relationships (typically 80-95%)
   - **Nearest-Neighbor Consistency**: Ranking preservation

## Usage

### Automatic Migration (Integrated)

The RAA server automatically detects dimension mismatches on startup and attempts migration:

```bash
# Just start the server - migration happens automatically
python src/server.py
```

**Server startup logs:**
```
INFO: ✓ Found pre-trained projection, migrating thought_nodes...
WARNING: ⚠ AUTO-MIGRATION: Automatic backup will be created.
INFO: ✓ Migration complete: 1,247 vectors migrated
INFO:   Backup: /path/to/chroma_backup_20231208_120000
INFO:   Semantic preservation: 87.3%
```

### Training Custom Projections

If no pre-trained projection exists for your model pair, train one:

```bash
python -m src.embeddings.migration_trainer \
  --source-model "BAAI/bge-large-en-v1.5" \
  --target-model "BAAI/bge-base-en-v1.5" \
  --samples 1000
```

**Options:**
- `--source-provider`: Source embedding provider (default: `sentence-transformers`)
- `--source-model`: Source model name (required)
- `--target-provider`: Target embedding provider (default: `sentence-transformers`)
- `--target-model`: Target model name (required)
- `--samples`: Number of training samples (default: 1000, more = better quality)
- `--output-dir`: Output directory for projection (default: `src/embeddings/projections`)
- `--device`: Training device (`cuda` or `cpu`, auto-detected by default)

**Example output:**
```
Training projection: BAAI/bge-large-en-v1.5 → BAAI/bge-base-en-v1.5
Generating 1000 sample texts...
Training: 100%|████████| 100/100 [00:15<00:00, 6.45it/s]
Training complete - Val Loss: 0.000123, Similarity Preservation: 92.4%

============================================================
Training Complete!
============================================================
Projection: src/embeddings/projections/BAAI_bge-large-en-v1.5_to_BAAI_bge-base-en-v1.5.pt
MSE: 0.000123
Similarity Preservation: 92.4%
============================================================
```

### Manual Migration (Python API)

```python
from src.embeddings.chroma_migrator import ChromaMigrator
from pathlib import Path

# Initialize migrator
migrator = ChromaMigrator("/path/to/chroma_data")

# Check for mismatches
mismatches = migrator.check_dimensions(expected_dim=1536)
print(f"Found {len(mismatches)} mismatched collections")

# Migrate a collection
result = migrator.migrate_collection(
    collection_name="thought_nodes",
    projection_path="src/embeddings/projections/bge-large_to_openai.pt",
    auto_backup=True,
    verify=True,
)

print(f"Migrated {result['n_migrated']} vectors")
print(f"Backup: {result['backup_path']}")
print(f"Similarity preservation: {result['verification']['similarity_correlation']:.2%}")
```

## Pre-trained Projections

Pre-trained projections are stored in `src/embeddings/projections/` and registered in `registry.json`.

### Available Projections

Currently, the system ships without pre-trained projections. You can train common model pairs:

**Popular Migrations:**
- BGE-large (1024D) → BGE-base (768D)
- BGE-large (1024D) → OpenAI Ada-002 (1536D)
- BGE-base (768D) → OpenAI text-embedding-3-small (1536D)

### Training for Popular Models

```bash
# BGE large to base
python -m src.embeddings.migration_trainer \
  --source-model "BAAI/bge-large-en-v1.5" \
  --target-model "BAAI/bge-base-en-v1.5" \
  --samples 2000

# BGE to OpenAI
python -m src.embeddings.migration_trainer \
  --source-model "BAAI/bge-large-en-v1.5" \
  --target-provider openai \
  --target-model "text-embedding-ada-002" \
  --samples 2000
```

## Architecture

```
src/embeddings/
├── migration.py              # Core projection engine
│   ├── ProjectionModel       # Linear transformation layer
│   ├── EmbeddingMigration    # Training & transformation
│   └── MigrationDetector     # Auto-detection
├── chroma_migrator.py        # Chroma-specific migration
│   └── ChromaMigrator        # Batch migration with backup
├── migration_trainer.py      # CLI training utility
├── projections/              # Pre-trained models
│   ├── registry.json         # Model metadata
│   └── *.pt                  # PyTorch projection weights
└── README.md                 # This file
```

## Technical Details

### Why This Works

- **Similar Semantic Spaces**: Modern embedding models learn similar semantic representations
- **Linear Transformations**: Preserve angular relationships (cosine similarity)
- **Fast**: Matrix multiplication is highly optimized (microseconds per vector)
- **Compact**: Projection models are ~1-10MB (vs. re-embedding gigabytes of data)

### Limitations

- **Best for similar models**: Works best between models of similar architecture/training
- **Quality degradation**: Expect 5-20% semantic drift (varies by model pair)
- **Dimensionality reduction**: Reducing dimensions (e.g., 1024D → 768D) loses some information
- **Not perfect**: Some edge cases may have higher accuracy loss

### When to Use

✅ **Good candidates:**
- Upgrading within a model family (BGE-large → BGE-base)
- Switching to better/faster models with similar training
- Large existing databases where re-embedding is expensive
- Testing new models without losing existing data

❌ **Poor candidates:**
- Drastically different models (word2vec ↔ transformers)
- Critical applications requiring 100% accuracy
- Very small databases (re-embedding is cheap)
- Semantic search where precision is paramount

## Best Practices

1. **Always backup**: Even though auto-backup is enabled, keep additional copies
2. **Train with more samples**: Use 2000-5000 samples for production migrations
3. **Verify after migration**: Check a sample of results to ensure quality
4. **Test first**: Migrate a copy of your database before production
5. **Monitor similarity preservation**: Aim for >85% for production use
6. **Document your migrations**: Record which projections were used and when

## Troubleshooting

### No pre-trained projection found

```
✗ No pre-trained projection found for 1024D → 1536D
```

**Solution**: Train a custom projection:
```bash
python -m src.embeddings.migration_trainer \
  --source-model <your_old_model> \
  --target-model <your_new_model> \
  --samples 2000
```

### Migration failed

```
✗ Migration failed for thought_nodes: <error>
```

**Solutions**:
1. Check the backup location (automatically created)
2. The database has been rolled back automatically
3. Try training with more samples
4. Consider fresh start if migration quality is poor

### Low similarity preservation

```
INFO: Semantic preservation: 65.2%
```

**If < 80%**:
- Train with more samples (--samples 5000)
- Verify source/target models are compatible
- Consider that some model pairs just don't map well

## Future Enhancements

- [ ] Non-linear projection options (neural networks)
- [ ] Multi-stage projection chains (A → B → C)
- [ ] Quality-based automatic projection selection
- [ ] Projection composition (combine multiple projections)
- [ ] Web UI for migration management
- [ ] More pre-trained common model pairs

## Contributing

To add a pre-trained projection to the registry:

1. Train the projection
2. The trainer automatically updates `registry.json`
3. Commit both the `.pt` file and updated registry
4. Document in this README under "Available Projections"

## References

- **Linear Transformations**: [Mikolov et al., 2013 - Exploiting Similarities among Languages for Machine Translation](https://arxiv.org/abs/1309.4168)
- **Embedding Space Properties**: [Levy & Goldberg, 2014 - Neural Word Embedding as Implicit Matrix Factorization](https://papers.nips.cc/paper/2014/hash/feab05aa91085b7a8012516bc3533958-Abstract.html)
- **Vector Database Migration**: Industry best practices from Pinecone, Weaviate, and MongoDB Atlas Vector Search

---

Built with ❤️ for the Reflective Agent Architecture (RAA)
