# Copilot Instructions: Reflective Agent Architecture (RAA)

## Project Overview

RAA is a research prototype implementing a metacognitive AI architecture that detects confusion through entropy monitoring and searches associative memory for alternative conceptual framings to achieve insight-like problem solving. The system bridges modern Hopfield networks, transformer attention, and entropy-based metacognition.

## Architecture: The "Aha!" Loop

Four core components compose the metacognitive loop:

1. **Manifold** (`src/manifold/`) - Modern Hopfield Network storing semantic knowledge as energy landscapes
2. **Processor** (`src/processor/`) - Transformer decoder for token-level generation
3. **Pointer** (`src/pointer/`) - Goal controller maintaining persistent goal state (RNN/SSM)
4. **Director** (`src/director/`) - Metacognitive monitor detecting entropy spikes and triggering associative search

**Critical flow:** Processor generates → Director monitors entropy → Clash detected → Director searches Manifold → Updates Pointer with new goal → Processor regenerates with new framing → Success

### The Three-Stage Tunneling Mechanism

RAA achieves insight through **energy-based stuck detection** followed by **three-stage tunneling** for escaping local basins:

**Stage 1: Local Escape (k-NN Search)**
- Uses k-nearest neighbors in embedding space
- Excludes current basin (similarity > 0.95 threshold)
- Handles: "Try related concept" (e.g., stuck on algebra → try geometry)
- Implementation: `energy_aware_knn_search()` in `src/director/search_mvp.py`

**Stage 2: Semantic Tunneling (Intent-Preserving)**
- Jumps to distant basins while preserving query intent
- Original query embedding = the intent to maintain
- Selects patterns: `distance_from_stuck > 0.3 AND relevance_to_intent > 0.7`
- Handles: Exploration without losing problem relevance
- Status: Planned for Phase 2

**Stage 3: Analogical Tunneling (Graph-Based)**
- Multi-hop reasoning through CWD knowledge graph
- Uses `cwd.hypothesize()` to find bridge concepts
- Enables creative leaps: stuck → bridge_1 → bridge_2 → insight
- Handles: Non-obvious connections, true insight moments
- Status: Planned for Phase 2 (CWD integration)

**"Aha!" Detection:**
```
High Energy (stuck: -2.0)
  → Tunneling Search (try stages 1→2→3)
  → New Basin Found (low energy: -8.0)
  → Energy Reduction > threshold
  → Success! (insight achieved)
```

### Pattern Association Mechanics

Patterns in Manifold form emergent associations through:
1. **Geometric proximity** - Cosine similarity in embedding space
2. **Beta-scaled attention** - `softmax(β * X^T ξ)` creates association strengths
3. **Retrieval dynamics** - Multi-step convergence strengthens connections

Example: Storing `["red", "apple", "fruit", "round", "sweet"]` creates graph:
```
red ←→ apple ←→ fruit
         ↓
       sweet
```
Query "red" retrieves ["apple" (high attention), "fruit" (medium)]

## Key Design Principles

### Energy-Based Retrieval
- Hopfield energy function: `E(ξ) = -lse_β(X^T ξ) + 0.5 ||ξ||²`
- Lower energy = more stable attractor basin
- Search uses energy-aware k-NN selection (not pure distance)
- **Always normalize embeddings** before storing/retrieving: `F.normalize(pattern, p=2, dim=-1)`

### Beta Scaling (CRITICAL)
- Beta controls attention sharpness in Hopfield networks
- **Requires ~10x range for meaningful entropy variation** (not 2x)
- Current validated range: `beta_min=5.0` to `beta_max=50.0` (increased from initial 0.5-2.0)
- Adaptive beta modulates exploration (low beta) vs exploitation (high beta)
- See `docs/BETA_SCALING_AND_TESTING.md` for empirical validation

### Two Integration Modes
1. **Generation Loop** (`ReflectiveAgentArchitecture`) - Token-based with actual Processor logits
2. **Reasoning Loop** (`RAAReasoningLoop`) - Pure embedding-based using pseudo-logits from attention distributions

**Testing philosophy:** Always test with complete RAA system (all four components), not isolated components. Pseudo-logits from embeddings don't reflect real entropy variance.

## Development Workflows

### Setup & Dependencies
```bash
# Installation (uv is the standard tool - ALWAYS use it)
uv sync                    # Basic install
uv sync --extra dev        # With pytest, black, ruff, mypy
uv sync --extra notebooks  # Add Jupyter support
uv sync --extra server     # CWD + Ollama MCP server (includes chromadb, neo4j, sentence-transformers, ollama)

# Environment
uv run python script.py    # Execute with managed environment
```

### MCP Server Setup (CWD Integration)
```bash
# 1. Configure environment variables
cp .env.example .env
# Edit .env: NEO4J_PASSWORD (required), NEO4J_URI, EMBEDDING_MODEL

# 2. Start external services (required for MCP server)
# Neo4j (Docker or Desktop):
docker run -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j:your_password neo4j
# Ollama:
ollama serve
ollama pull qwen2.5:3b  # or qwen3:latest

# 3. Run MCP server (stdio mode for Claude Desktop)
uv run python src/server.py

# 4. Configure Claude Desktop (see examples/cwd_integration_example.py for config)
```

### Testing Commands
```bash
# Quick validation
uv run python validate_energy_search.py

# Full test suite
uv run pytest -v
uv run pytest --cov=src --cov-report=html

# Component-specific tests
uv run pytest tests/test_director.py::test_energy_aware_search -v
uv run pytest tests/test_manifold.py -v

# Integration tests (Phase 1 CWD-RAA bridge)
uv run pytest tests/test_integration_phase1.py -v
```

### Evaluation & Benchmarks
```bash
# Remote Associates Test (RAT) - main insight task
uv run python experiments/insight_tasks/run_rat_evaluation.py

# Full system test with token generation (recommended for testing beta scaling)
uv run python examples/full_system_generation_test.py

# Baseline comparison
uv run python experiments/baselines/transformer_baseline.py

# Complete benchmark suite
python experiments/run_benchmark.py --mode full --verbose
```

### Data Requirements
- **GloVe embeddings** needed for RAT evaluation: `data/embeddings/glove.6B.100d.txt`
- Download from Stanford NLP if missing
- Loaded via `src/manifold/glove_loader.py`

## Project-Specific Conventions

### Configuration Pattern
All components use dataclass configs, not constructor kwargs:
```python
from src.manifold import HopfieldConfig, Manifold

config = HopfieldConfig(
    embedding_dim=256,
    beta=10.0,
    adaptive_beta=True,
    device="cpu"
)
manifold = Manifold(config)
```

### Entropy Monitoring
- Shannon entropy computed from logits: `H = -Σ p_i log p_i`
- **Director uses adaptive thresholds** (75th percentile of history)
- Clash detection triggers when `entropy > threshold`
- Track entropy per-step in reasoning loops for diagnostics

### Search Mechanism
- Primary: `energy_aware_knn_search()` - combines distance with Hopfield energy
- Fallback: `knn_search()` - pure k-NN
- Excludes current basin via similarity threshold (default 0.95)
- Returns `SearchResult` with best pattern, neighbors, and search metrics

**Energy-based stuck detection:**
- Low energy (e.g., -8.0) = "In stable basin, keep going"
- High energy (e.g., -2.0) = "Stuck between basins, trigger tunneling"
- Direct measure of attractor basin stability (no dimensionality issues)
- Beta-responsive: Sharp beta → narrow basins, flat beta → wide basins

### Reframing Control
- `max_reframing_attempts` prevents infinite loops (default: 3)
- Count reframing episodes as diagnostic metric
- RAT evaluation shows ~30.5 reframings per item on average
- Zero reframings suggests threshold/beta misconfiguration

## Common Pitfalls

1. **Testing with pseudo-logits only** - Use full system test with Processor for realistic entropy variance
2. **Beta range too narrow** - Needs 10x range (5.0-50.0), not 2x (0.5-2.0)
3. **Forgetting normalization** - Always normalize embeddings before Manifold operations: `F.normalize(pattern, p=2, dim=-1)`
4. **Empty entropy history** - Check for NaN handling when computing statistics on empty lists
5. **Wrong integration mode** - Generation tasks need `ReflectiveAgentArchitecture`, embedding tasks need `RAAReasoningLoop`
6. **Hardcoding credentials** - Never hardcode Neo4j/Ollama credentials; always use environment variables (`.env` file)
7. **Missing external services** - MCP server requires Neo4j (bolt://localhost:7687) and Ollama (localhost:11434) running
8. **Not using dataclass configs** - All components use `@dataclass` configs, never positional args in constructors
9. **Skipping CWD deconstruction** - For context-dependent queries, always `deconstruct` source material first before synthesis

## Key Files for Understanding System

- `src/integration/raa_loop.py` - Complete RAA orchestration (`ReflectiveAgentArchitecture` class)
- `src/director/director_core.py` - Entropy monitoring + search logic
- `src/manifold/hopfield_network.py` - Energy function and retrieval dynamics
- `examples/full_system_generation_test.py` - Proper testing approach with real logits
- `docs/INTEGRATION_ARCHITECTURE.md` - Component composition design rationale
- `docs/BETA_SCALING_AND_TESTING.md` - Empirical validation of parameter ranges

## Current Performance Baselines

- **RAT Accuracy**: 20% (RAA) vs 0% (no reframing baseline)
- **Reframing Frequency**: ~30 search episodes per problem
- **Processing Speed**: 0.16s per item (50 reasoning steps)
- **Improvement Opportunity**: Learned embeddings, neural network logit generation, fine-tuned Processor

## Integration Roadmap (CWD-RAA Bridge)

**Phase 1 (40% complete):** Infrastructure - embedding mapper, entropy calculator, monitoring
**Phase 2:** Entropy-triggered search integration
- Implement semantic tunneling (intent-preserving distant jumps)
- Integrate CWD `hypothesize()` for analogical tunneling
- Add tunneling stage fallback logic (k-NN → semantic → analogical)

**Phase 3:** Utility-biased energy function
**Phase 4:** Bidirectional learning with compression-based reinforcement

**Why RAA + CWD integration:**
- **RAA**: Fast, local, energy-based (focused search in continuous space)
- **CWD**: Deep, global, graph-based (intuitive leaps through knowledge topology)
- **Together**: Local hill-climbing + occasional long jumps = human-like insight

See `src/integration/README.md`, `docs/INTEGRATION_PROGRESS.md`, and `src/integration/cwd_raa_bridge.py` for details.

### Integration Tip: Exposing Goal State to the Bridge
- Pass a `Pointer` (GoalController) instance to `CWDRAABridge(pointer=pointer)` or a callback `get_current_goal=lambda: pointer.get_current_goal()`.
- On entropy clash, the bridge can call `Director.search(current_goal)` and, if a `Pointer` is provided, update the goal via `pointer.set_goal(result.best_pattern)`.

## MCP Server Architecture (System 2 + System 3)

### System 2: RAA-CWD Reasoning Loop
The MCP server (`src/server.py`) integrates two cognitive systems:
- **RAA (Reflective Agent Architecture)**: Fast metacognitive monitoring with entropy-driven search
- **CWD (Cognitive Workspace Database)**: Graph-based reasoning with topology tunneling

**Key MCP Tools**:
- `deconstruct`: Break problems into State/Agent/Action fragments (Tripartite Manifold)
- `hypothesize`: Find analogical bridges between concepts via topology tunneling
- `synthesize`: Merge thought-nodes in latent space with compression tracking
- `revise`: Continuous belief revision using Hybrid Operator C (LTN + Hopfield)
- `check_cognitive_state`: Monitor agent's thinking mode (Focused/Broad/Looping)

### System 3: Escalation Architecture
When internal search exhausts (high entropy + max search attempts), Director escalates to:
- **COMPASS Framework**: Multi-step planning and metacognitive orchestration (SHAPE → oMCD → SLAP → SMART)
- **Heavy-Duty Models**: External consultation (Claude Opus, o1) via `consult_compass` tool
- **Antifragile Agents**: Specialized ephemeral agents spawned for topological obstructions:
  - **Debater Agents** for tension loops (contradictions)
  - **Explorer Agents** for H¹ holes (missing concepts)

### Cognitive State Monitoring
The Director tracks:
- **Entropy**: Shannon entropy from logits/attention distributions
- **Energy**: Hopfield energy (basin stability)
- **oMCD**: Cognitive resource allocation (0-100 scale)
- **Cognitive Mode**: Focused/Broad/Looping/Unknown

**Automatic interventions**:
- Entropy > 80.0 → Triggers COMPASS consultation
- oMCD > 80.0 → Resource allocation warning
- "Looping" state → Diagnose topological obstructions

See `RAA_AGENT.md` for complete agent protocol and `docs/SYSTEM3_ESCALATION_ARCHITECTURE.md` for escalation design.

## Additional Resources

- Theoretical foundations: `docs/REFERENCES.md` (Nobel Prize-winning Modern Hopfield Networks, entropy-based metacognition papers)
- Search design: `docs/SEARCH_MECHANISM_DESIGN.md`
- System 3 escalation: `docs/SYSTEM3_ESCALATION_ARCHITECTURE.md`
- Agent protocol: `RAA_AGENT.md` (operational guide for AI agents using MCP server)
- CWD integration: `src/integration/README.md` (Phase 1-4 roadmap)
- Known issues: `todo_list.md`
