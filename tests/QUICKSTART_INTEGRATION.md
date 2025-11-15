# Quick Start: RAA Integration Layer

## What Was Built

The **integration layer** that makes all four RAA components work together:

```
Manifold (Memory) + Director (Monitor) + Pointer (Goal) → Working System
```

## Files Created

1. **`src/integration/reasoning_loop.py`** - Core integration logic
2. **`tests/test_integration.py`** - Integration tests
3. **`examples/simplified_rat_solver.py`** - Working demo
4. **`docs/INTEGRATION_ARCHITECTURE.md`** - Design document

## Quick Test

### 1. Run Integration Tests
```bash
cd /home/ty/Repositories/ai_workspace/reflective-agent-architecture

# Option 1: Using uv
uv run python -m pytest tests/test_integration.py -v

# Option 2: Using python directly (if venv exists)
python -m pytest tests/test_integration.py -v
```

### 2. Run the Demonstration
```bash
# This shows the integration layer solving RAT problems
uv run python examples/simplified_rat_solver.py

# Or directly:
python examples/simplified_rat_solver.py
```

Expected output: Shows reasoning steps, reframing events, and energy trajectories.

## What the Integration Does

### Before Integration
```
✗ Manifold works → but how to use it?
✗ Director works → but how to connect?
✗ Pointer works → but how to update?
✗ Components isolated, no composition
```

### After Integration
```
✓ RAAReasoningLoop composes all components
✓ Clear data flow: embeddings → goal → search → solution
✓ Metrics tracking throughout
✓ Working proof-of-concept
```

## Example Usage

```python
from src.manifold import Manifold, HopfieldConfig
from src.director import Director, DirectorConfig
from src.pointer import GoalController, PointerConfig
from src.integration import RAAReasoningLoop, ReasoningConfig

# Initialize components
manifold = Manifold(config=HopfieldConfig(embedding_dim=256))
director = Director(manifold=manifold, config=DirectorConfig())
pointer = GoalController(config=PointerConfig(embedding_dim=256))

# Create integration loop
loop = RAAReasoningLoop(
    manifold=manifold,
    director=director,
    pointer=pointer,
    config=ReasoningConfig(max_steps=20)
)

# Solve a problem
import torch
problem_embedding = torch.randn(256)
solution, metrics = loop.reason(input_embeddings=problem_embedding)

print(f"Solved in {metrics['total_steps']} steps")
print(f"Reframings: {metrics['num_reframings']}")
print(f"Final energy: {metrics['final_energy']:.4f}")
```

## Next Steps

### 1. Verify Integration Works
```bash
# Run tests - should all pass
python -m pytest tests/test_integration.py -v
```

### 2. Run Demonstration
```bash
# See the integration in action
python examples/simplified_rat_solver.py
```

### 3. Move to Phase 2: Empirical Validation

**What's needed:**
- Pre-trained embeddings (BERT/GPT instead of random)
- Full RAT dataset with ground truth
- Baseline comparison (RAA vs no-Director)

See `INTEGRATION_LAYER_SUMMARY.md` for complete roadmap.

## Troubleshooting

### If pytest not found:
```bash
# Install pytest in venv
uv pip install pytest

# Or use:
uv sync --extra dev
```

### If imports fail:
```bash
# Make sure you're in project root
cd /home/ty/Repositories/ai_workspace/reflective-agent-architecture

# Run with python path
PYTHONPATH=. python examples/simplified_rat_solver.py
```

### If torch not found:
```bash
# Sync dependencies
uv sync

# Verify installation
uv run python -c "import torch; print(torch.__version__)"
```

## Architecture Summary

```
Input Embedding
      ↓
[Pointer: Initialize goal]
      ↓
[Manifold: Retrieve via energy minimization]
      ↓
[Compute pseudo-logits for entropy]
      ↓
[Director: Monitor entropy, trigger search if clash]
      ↓
[Pointer: Update goal with search result]
      ↓
Repeat until convergence or max steps
      ↓
Solution Embedding + Metrics
```

**Key Innovation**: Pseudo-logits strategy bridges embedding and entropy spaces.

## Files Modified

**Created (6 files)**:
- `src/integration/reasoning_loop.py` (340 lines)
- `src/integration/__init__.py`
- `tests/test_integration.py` (350 lines)
- `examples/simplified_rat_solver.py` (330 lines)
- `docs/INTEGRATION_ARCHITECTURE.md` (full design spec)
- `INTEGRATION_LAYER_SUMMARY.md` (this summary)

**Updated (2 files)**:
- `README.md` (implementation plan updated)
- Various __init__.py exports

Total: ~1200 lines of integration code + comprehensive documentation

## Status

✅ **Phase 1.5 COMPLETE** - All components implemented and fixed  
✅ **Integration Layer COMPLETE** - Components compose correctly  
⏳ **Phase 2 READY** - Awaiting empirical validation with real embeddings

Next: Integrate pre-trained embeddings and run full RAT evaluation
