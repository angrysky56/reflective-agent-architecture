# Embedding Forensics

Identify unknown embedding models in your Chroma database.

## The Problem

Your Chroma collections might have embeddings from multiple different models:
- Previous experiments with different models
- Mixed data from different time periods
- Collections without metadata
- Legacy data before metadata tracking

**This tool identifies which models created which embeddings**, even when metadata is missing!

## How It Works

### 1. Dimension Analysis
Different models → different dimensions:
```
384D  → MiniLM-L6 models
768D  → BGE-base, MPNet-base
1024D → BGE-large, GTR-T5-large
1536D → OpenAI Ada-002, text-embedding-3-small
```

### 2. Metadata Extraction
Checks collection metadata for:
- `embedding_model`
- `embedding_provider`
- `embedding_dim`

### 3. Statistical Fingerprinting (Future)
Analyzes vector properties:
- Mean, std, norm distribution
- Clustering patterns
- Sparsity metrics

## Usage

### CLI - Scan All Collections

```bash
python -m src.embeddings.forensics /path/to/chroma_data
```

**Output:**
```
🔍 Scanning collection: thought_nodes
Total vectors: 1,247

📊 Dimension: 1024D (800 vectors)
  ✓ Detected from metadata: BAAI/bge-large-en-v1.5

📊 Dimension: 1536D (447 vectors)
  ⚠ No metadata found
  Possible models:
    1. openai/text-embedding-ada-002
    2. openai/text-embedding-3-small

✓ Analysis complete! Report saved to: embedding_forensics_report.json
```

### CLI - Scan Specific Collection

```bash
python -m src.embeddings.forensics /path/to/chroma_data \
  --collection thought_nodes
```

### CLI - Interactive Mode

```bash
python -m src.embeddings.forensics /path/to/chroma_data \
  --interactive
```

Prompts you to identify unknown models:
```
Possible models:
  1. openai/text-embedding-ada-002
  2. openai/text-embedding-3-small
Select model (1-2) or type custom: 1
```

### Python API

```python
from src.embeddings.forensics import EmbeddingForensics

# Initialize
forensics = EmbeddingForensics("/path/to/chroma_data")

# Scan collection
results = forensics.scan_collection("thought_nodes")

# Interactive resolution
resolved_models = forensics.interactive_resolve(results)

# Generate migration plan
plan = forensics.generate_migration_plan(results, target_dim=1536)

# Save report
forensics.save_forensics_report(results, "report.json")
```

## Forensics Report Format

```json
{
  "status": "analyzed",
  "collection": "thought_nodes",
  "total_count": 1247,
  "sampled": 100,
  "dimension_groups": {
    "1024": {
      "dimension": 1024,
      "count": 800,
      "models_from_metadata": ["BAAI/bge-large-en-v1.5"],
      "candidate_models": ["BAAI/bge-large-en-v1.5", "..."],
      "confidence": "high"
    },
    "1536": {
      "dimension": 1536,
      "count": 447,
      "models_from_metadata": [],
      "candidate_models": ["openai/text-embedding-ada-002", "..."],
      "confidence": "low"
    }
  }
}
```

## Migration Workflow

### 1. Scan Database

```bash
python -m src.embeddings.forensics /path/to/chroma_data --interactive
```

### 2. Review Report

Check `embedding_forensics_report.json`:
- Which dimensions are present?
- How many vectors per dimension?
- Which models were identified?

### 3. Train Projections (if needed)

For each detected model → target model:

```bash
python -m src.embeddings.migration_trainer \
  --source-model "BAAI/bge-large-en-v1.5" \
  --target-model "openai/text-embedding-3-small" \
  --samples 2000
```

### 4. Migrate

Restart server - automatic migration will handle it!

Or manually:

```python
from src.embeddings.chroma_migrator import ChromaMigrator

migrator = ChromaMigrator("/path/to/chroma_data")
migrator.migrate_collection(
    collection_name="thought_nodes",
    projection_path="projections/bge-large_to_openai.pt",
    auto_backup=True
)
```

## Adding New Models

To add support for new models, edit `forensics.py`:

```python
KNOWN_MODELS = {
    384: [...],
    768: [...],
    1024: [...],
    1536: ["openai/text-embedding-ada-002", "YOUR_NEW_MODEL"],  # Add here
    # Add new dimensions as needed
}
```

## Limitations

- **Dimension overlap**: Some models share dimensions (e.g., BGE-large and Cohere both use 1024D)
- **No metadata**: Requires user input for confirmation when metadata is missing
- **Statistical analysis**: Advanced fingerprinting not yet implemented
- **Performance**: Scans sample of collection (not all vectors)

## Future Enhancements

- [ ] Advanced statistical fingerprinting
- [ ] Machine learning model identification
- [ ] Automatic projection training for detected models
- [ ] Split collections by detected model
- [ ] Parallel collection scanning
- [ ] Web UI for forensics analysis

## Tips

1. **Always scan before migrating** - Know what you're working with
2. **Use interactive mode first time** - Confirm model identifications
3. **Keep reports** - Track migration history
4. **Sample size matters** - Larger samples = more confident detection
5. **Document your models** - Add notes to forensics reports

---

Built to solve the "mixed embeddings" problem in production vector databases!
