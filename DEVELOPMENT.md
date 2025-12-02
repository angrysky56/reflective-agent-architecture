# RAA Development & Architecture Guide

This document contains detailed technical information, theoretical foundations, and development instructions for the Reflective Agent Architecture (RAA).

## Theoretical Foundations

RAA synthesizes three active research frontiers:
1. **Modern Hopfield Networks**: Exponential-capacity associative memory (2024 Nobel Prize in Physics context).
2. **Entropy-Based Metacognition**: Detection of model uncertainty through predictive entropy monitoring.
3. **Sheaf Cohomology**: Topological analysis of network learning dynamics via cellular sheaf theory.

## Core Components

### 1. The Tripartite Manifold (Associative Memory)
- **Implementation**: Three specialized Modern Hopfield Networks (vmPFC, amPFC, dmPFC).
- **Function**: Stores semantic knowledge as energy landscapes in orthogonal domains:
  - **State (vmPFC)**: Static context (Low Beta).
  - **Agent (amPFC)**: Personas and intent (Medium Beta).
  - **Action (dmPFC)**: Transition dynamics (High Beta).

### 2. The Director (Metacognitive Monitor)
- **Function**: Detects "clashes" (high-entropy states) and triggers search.
- **Mechanism**: Calculates Shannon entropy on Transformer output. If entropy > threshold, it suppresses the current goal and searches the Manifold for alternative framings.

### 3. The Precuneus (Integrator)
- **Function**: Fuses the three Manifold streams into a unified experience.
- **Mechanism**: Uses Energy Gating to silence high-energy confusion and a Continuity Field to maintain identity coherence over time.

### 4. System 3: Adaptive Agents & COMPASS
- **Adaptive Agents**: Ephemeral agents spawned to resolve topological obstructions (e.g., "Debater" for tension loops).
- **COMPASS Integration**: Bidirectional interoperability allowing RAA to delegate complex planning tasks to the COMPASS framework via `consult_compass`.

### 5. Operator C (Belief Revision)
- **Tool**: `revise`
- **Mechanism**: Hybrid search combining Logic Tensor Networks (LTN) truth values and Hopfield energy minimization to refine beliefs against evidence and constraints.

## Development Setup

### Prerequisites
- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) (recommended)

### Installation
```bash
# Clone and sync dependencies
git clone https://github.com/angrysky56/reflective-agent-architecture.git
cd reflective-agent-architecture
uv sync --extra dev --extra server
```

## Running Tests

```bash
# Run full test suite
uv run pytest

# Run specific energy-aware tests
uv run pytest tests/test_director.py::test_energy_aware_search -v

# Run Remote Associates Test (RAT) evaluation
uv run python experiments/insight_tasks/run_rat_evaluation.py

# Run full system generation test
uv run python examples/full_system_generation_test.py
```

## Benchmarks & Results
- **RAA System**: ~20.0% accuracy on RAT tasks (vs 0% baseline).
- **Reframing**: Averages 30.5 search episodes per item.
- **Beta Scaling**: Adaptive beta range (5.0-50.0) enables meaningful entropy modulation.

## Project Structure
```
reflective-agent-architecture/
├── src/
│   ├── manifold/           # Hopfield Networks
│   ├── director/           # Metacognitive monitor
│   ├── integration/        # RAA-CWD bridge & server
│   └── server.py           # MCP Server entry point
├── tests/                  # Unit tests
├── docs/                   # Documentation
└── experiments/            # Benchmarks
```
