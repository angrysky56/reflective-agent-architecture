# RAA-CWD Integration Module

**Status**: Phase 2 - Entropy-Triggered Search (Functional)
**Version**: 0.2.0

## Overview

This module bridges **Reflective Agent Architecture (RAA)** and **Cognitive Workspace Database (CWD)** to create a unified system for confusion-triggered utility-guided reframing.

## Quick Start with Real CWD (Ollama)

```bash
# 1. Set up environment
cp .env.example .env
# Edit .env and set NEO4J_PASSWORD

# 2. Install CWD dependencies
pip install ollama neo4j chromadb sentence-transformers python-dotenv

# 3. Start services
# - Ollama: ollama serve
# - Neo4j: docker run -p 7687:7687 -p 7474:7474 neo4j
# - Pull models: ollama pull qwen2.5:3b && ollama pull nomic-embed-text

# 4. Run integration example
python examples/cwd_integration_example.py
```

See `examples/cwd_integration_example.py` for complete working code.

## Architecture

```
┌─────────────────────────────────────────────┐
│           Integration Module                │
├─────────────────────────────────────────────┤
│                                              │
│  ┌──────────────┐      ┌─────────────────┐ │
│  │    Bridge    │◄────►│ Embedding Mapper│ │
│  │ (Orchestrator)│      │  (Space Align)  │ │
│  └──────┬───────┘      └─────────────────┘ │
│         │                                    │
│         ├──► EntropyCalculator              │
│         ├──► UtilityAwareSearch (Phase 3)   │
│         └──► AttractorReinforcement (Phase 4)│
│                                              │
└─────────────────────────────────────────────┘
         │              │
    ┌────▼────┐    ┌───▼────┐
    │   RAA   │    │  CWD   │
    │Director │    │ Server │
    │Manifold │    │  Tools │
    └─────────┘    └────────┘
```

## Components

### 1. CWDRAABridge
**Location**: `cwd_raa_bridge.py`
**Status**: ✅ Phase 2 Complete (search + goal updates)

Main orchestrator that:
- Executes CWD operations with RAA monitoring
- Detects entropy spikes using Director's adaptive thresholds
- Triggers RAA search on confusion
- Updates Pointer goal with search results
- Tracks integration metrics

**Example Usage**:
```python
from src.integration import CWDRAABridge

bridge = CWDRAABridge(
    cwd_server=cwd_server,
    raa_director=director,
    manifold=manifold,
    pointer=pointer,  # Optional: enables goal updates
    raa_director=director,
    manifold=manifold
)

# Execute monitored operation
result = bridge.execute_monitored_operation(
    operation='hypothesize',
    params={'node_a_id': 'n1', 'node_b_id': 'n2'}
)

# Check metrics
metrics = bridge.get_metrics()
print(f"Operations monitored: {metrics['operations_monitored']}")
print(f"Entropy spikes: {metrics['entropy_spikes_detected']}")
```

### 2. EmbeddingMapper
**Location**: `embedding_mapper.py`
**Status**: ✅ Fully Implemented

Converts between CWD's graph space and RAA's Hopfield space:
- CWD nodes → embedding vectors
- CWD tools → embedding vectors
- Similarity computation
- Dimension handling

**Example Usage**:
```python
from src.integration import EmbeddingMapper

mapper = EmbeddingMapper(embedding_dim=512)

# Convert node
vector = mapper.cwd_node_to_vector(
    node_id="n1",
    node_content="This is a thought about consciousness",
    node_metadata={"goal": "understand_mind"}
)

# Convert tool
tool_vector = mapper.tool_to_vector(
    tool_id="lock_and_key",
    tool_description="Apply lever mechanism to constraints",
    tool_use_cases=["locks", "latches"]
)

# Compute similarity
similarity = mapper.compute_similarity(vector, tool_vector)
```

**Dependencies**: Requires `sentence-transformers`
```bash
pip install sentence-transformers
```

### 3. EntropyCalculator
**Location**: `entropy_calculator.py`
**Status**: ✅ Fully Implemented

Converts CWD operation results to entropy signals:
- Hypothesize → confidence distribution
- Synthesize → quality distribution
- Constrain → satisfaction distribution
- Shannon entropy computation

**Example Usage**:
```python
from src.integration import EntropyCalculator, cwd_to_logits

calculator = EntropyCalculator(temperature=1.0)

# From hypothesize operation
hypotheses = [
    {"hypothesis": "H1", "confidence": 0.9},
    {"hypothesis": "H2", "confidence": 0.3}
]

logits = calculator.hypothesize_to_logits(hypotheses)
entropy = calculator.compute_entropy(logits)

print(f"Entropy: {entropy:.3f} bits")  # Low entropy = confident

# Convenience function
logits = cwd_to_logits('hypothesize', hypotheses)
```

### 4. UtilityAwareSearch
**Location**: `utility_aware_search.py`
**Status**: 🔴 Phase 3 Placeholder

Will implement utility-biased energy function:
```
E_biased(ξ) = E_hopfield(ξ) - λ * U(tool)
```

### 5. AttractorReinforcement
**Location**: `reinforcement.py`
**Status**: 🔴 Phase 4 Placeholder

Will implement compression-based attractor strengthening.

## Testing

### Run Integration Tests
```bash
# Install dependencies
pip install sentence-transformers pytest pytest-cov

# Run Phase 1 tests
pytest tests/test_integration_phase1.py -v

# Run with coverage
pytest tests/test_integration_phase1.py --cov=src.integration
```

### Test Coverage
- ✅ Embedding mapper (round-trip, similarity)
- ✅ Entropy calculator (all operations)
- ✅ Bridge initialization
- ✅ Monitored execution
- ✅ Metrics tracking

## Configuration

### Bridge Configuration
```python
from src.integration import BridgeConfig

config = BridgeConfig(
    # Embedding settings
    embedding_dim=512,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",

    # Monitoring
    entropy_threshold=2.0,
    enable_monitoring=True,

    # Search
    max_search_attempts=3,
    search_on_confusion=True,  # Phase 2

    # Metrics
    log_integration_events=True,
    device="cpu"
)

bridge = CWDRAABridge(cwd_server, director, manifold, config)
```

## Integration Phases

### ✅ Phase 1: Infrastructure (Current)
- [x] Embedding conversion
- [x] Entropy calculation
- [x] Basic monitoring
- [ ] Tool-pattern mapping
- [ ] Persistent storage

### 🔄 Phase 2: Entropy-Triggered Search
- [ ] Trigger RAA search on high entropy
- [ ] Route alternatives to CWD
- [ ] Integrated reasoning workflow

### 🔴 Phase 3: Utility-Biased Search
- [ ] Utility-aware energy function
- [ ] CWD goal integration
- [ ] Parameter tuning

### 🔴 Phase 4: Bidirectional Learning
- [ ] Compression-based reinforcement
- [ ] Attractor decay
- [ ] Meta-learning metrics

## Metrics Dashboard

The bridge tracks:
```python
metrics = {
    "operations_monitored": int,        # Total CWD ops monitored
    "entropy_spikes_detected": int,     # High entropy events
    "searches_triggered": int,          # RAA searches triggered (Phase 2)
    "alternatives_found": int,          # Successful alternatives (Phase 2)
    "integration_events": list[dict],   # Detailed event log
}
```

## Performance

### Expected Overhead
- Embedding computation: ~10ms per operation
- Entropy calculation: <1ms
- Monitoring overhead: <15ms total (negligible)

### Optimization Tips
1. **Batch operations**: Process multiple nodes together
2. **Cache embeddings**: Store frequently used vectors
3. **GPU acceleration**: Use `device="cuda"` for large batches

## Troubleshooting

### Common Issues

**1. Import Error: sentence_transformers**
```bash
pip install sentence-transformers
```

**2. Dimension Mismatch**
- Ensure `embedding_dim` matches RAA Manifold dimension
- EmbeddingMapper auto-resizes but logs warning

**3. Low Entropy Always**
- Check CWD operation results have variance
- Tune `temperature` parameter in EntropyCalculator

**4. Mock Data in Production**
- Replace `bridge._execute_cwd_operation()` with real CWD calls
- See Phase 1 completion tasks

## Dependencies

### Required
```
torch>=2.0.0
sentence-transformers>=2.0.0
```

### Optional
```
pytest>=7.0.0              # For testing
pytest-cov>=4.0.0          # For coverage
```

## Documentation

- **Design**: `/docs/RAA_CWD_INTEGRATION_DESIGN.md`
- **Progress**: `/docs/INTEGRATION_PROGRESS.md`
- **Session Summary**: `/docs/SESSION_SUMMARY_2025-01-16.md`
- **Adaptive Beta**: `/docs/ADAPTIVE_BETA_IMPLEMENTATION.md`

## Contributing

This module is under active development. Phase 1 (infrastructure) is ~40% complete.

**Current Focus**:
- Tool-pattern bidirectional mapping
- Persistent storage with SQLite
- Integration with real CWD server

**Next Phase**: Entropy-triggered search (Weeks 3-4)

## License

Part of the Reflective Agent Architecture project.

---

**Questions?** Check the documentation in `/docs/` or run the tests with `-v` for detailed output.
