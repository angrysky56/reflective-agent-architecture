# Sheaf Diagnostics for RAA

## Overview

The Sheaf Diagnostics module provides **topological analysis of predictive coding networks** based on the paper ["Sheaf Cohomology of Linear Predictive Coding Networks" (Seely, 2025)](https://arxiv.org/abs/2511.11092).

This module complements the existing entropy-based stuck-state detection with **principled mathematical tools** for understanding when and why learning may fail.

## Key Concepts

### 1. Cellular Sheaf Structure

A linear predictive coding network is formalized as a **cellular sheaf** where:
- **Vertices** (nodes) hold activation vectors
- **Edges** hold prediction error vectors
- **Restriction maps** are the network weights

The **coboundary operator** δ⁰ computes all prediction errors simultaneously from activations.

### 2. Cohomology Groups

- **H⁰**: Activation patterns with zero prediction error everywhere (typically trivial for clamped systems)
- **H¹**: **Irreducible error patterns** that cannot be eliminated by any choice of internal activations

> **Key Insight**: If dim(H¹) > 0, there exist prediction errors that inference CANNOT remove, regardless of how you optimize internal states.

### 3. Hodge Decomposition

For a supervised system with target error `b`, the Hodge decomposition splits:

```
b = (-D @ z*) + r*
     ────────   ──
     eliminable  harmonic
     by inference residual
```

Where:
- **H = I - D @ D†**: Harmonic projector (extracts irreducible errors)
- **G = D†**: Diffusive operator (determines optimal activations)

### 4. Monodromy Analysis

For networks with **feedback loops**, the monodromy Φ = W_feedback @ W_forward determines:

| Topology | Eigenvalues | Behavior |
|----------|-------------|----------|
| **Resonance** | Φ ≈ I | Slow inference, learnable |
| **Tension** | Φ ≈ -I | Internal contradictions, learning stalls |
| **Mixed** | Other | Depends on spectrum |

## Integration with RAA

### Combined Stuck-State Detection

```python
from src.director import Director, SheafAnalyzer

# Standard entropy check
is_clash, entropy = director.check_entropy(logits)

# Topological analysis
diagnosis = sheaf_analyzer.full_diagnosis(weights)

# Combined decision
should_escalate = (
    is_clash and 
    diagnosis.escalation_recommended
)
```

### Escalation Criteria

The sheaf diagnostics provide **principled escalation criteria**:

1. **Non-trivial H¹**: Irreducible errors exist → escalate
2. **Tension monodromy**: Feedback creates contradictions → escalate
3. **Low harmonic-diffusive overlap**: Learning is starved → escalate

### Key Metrics

| Metric | Meaning | Action |
|--------|---------|--------|
| `h1_dimension > 0` | Irreducible errors exist | Consider escalation |
| `harmonic_diffusive_overlap < 0.1` | Learning starved | Restructure or escalate |
| `topology == TENSION` | Feedback contradicts | Adjust initialization |

## Usage Examples

### Basic Diagnosis

```python
from src.director import SheafAnalyzer, SheafConfig

analyzer = SheafAnalyzer(SheafConfig(device="cpu"))

# Network weights
weights = [W1, W2, W3]

# Run diagnosis
diagnosis = analyzer.full_diagnosis(weights, target_error=b)

print(f"H^1 dimension: {diagnosis.cohomology.h1_dimension}")
print(f"Can fully resolve: {diagnosis.cohomology.can_fully_resolve}")
print(f"Escalation recommended: {diagnosis.escalation_recommended}")
```

### Monodromy Analysis

```python
# Analyze feedback loop
monodromy = analyzer.analyze_monodromy(W_forward, W_feedback)

if monodromy.topology == CognitiveTopology.TENSION:
    print("WARNING: Feedback creates internal contradictions")
    print("Consider adjusting weight initialization")
```

### Attention Pattern Analysis

```python
from src.director import AttentionSheafAnalyzer

attn_analyzer = AttentionSheafAnalyzer()

# Analyze attention weights from GoalBiasedAttention
results = attn_analyzer.diagnose_attention(attention_weights)

for head in results["per_head"]:
    if not head["can_learn"]:
        print(f"Head {head['head']} may have learning issues")
```

## Connection to RAA Architecture

### System 1-2-3 Integration

```
System 1 (Processor)
    │
    ▼ generates logits + attention
System 2 (Director)
    ├─ EntropyMonitor: detect high entropy
    │
    ├─ SheafAnalyzer: detect topological obstructions  ← NEW
    │
    └─ Search: find alternative goals
    │
    ▼ escalation criteria met?
System 3 (Escalation Manager)
    └─ External heavy compute
```

### The Escalation Decision Tree

```
High Entropy?
    │
    ├─ NO → Continue normal generation
    │
    └─ YES → Check Sheaf Diagnostics
              │
              ├─ H¹ = 0, topology OK → Try internal search
              │                         │
              │                         ├─ Found alternative → Use it
              │                         │
              │                         └─ Search exhausted → Escalate
              │
              └─ H¹ > 0 or TENSION → Escalate immediately
```

## Theoretical Background

### Why Sheaf Theory?

Sheaf theory provides a **local-to-global framework** that matches predictive coding's architecture:

- **Local consistency**: Each layer tries to minimize its prediction error
- **Global coherence**: Can local solutions assemble into a global solution?
- **Obstructions**: H¹ measures exactly what prevents this assembly

### The Harmonic-Diffusive Overlap

From Eq. 19 in Seely (2025):

```
∂E/∂W_e = (Hb)_e @ (Gb)_u^T
          ────────   ────────
          harmonic   diffusive
          (edge)     (vertex)
```

Learning requires **both**:
1. Non-zero harmonic residual (error signal exists)
2. Non-zero diffusive activation (source is active)

When these don't overlap, **learning is starved** even though both exist separately.

## File Structure

```
src/director/
├── __init__.py              # Exports sheaf diagnostics
├── director_core.py         # Main Director
├── entropy_monitor.py       # Entropy-based detection
├── search_mvp.py            # k-NN search
└── sheaf_diagnostics.py     # Sheaf analysis (NEW)

tests/
└── test_sheaf_diagnostics.py  # 18 tests

examples/
└── sheaf_diagnostics_demo.py  # Comprehensive examples
```

## References

- **Primary**: Seely, J. (2025). "Sheaf Cohomology of Linear Predictive Coding Networks". arXiv:2511.11092
- **Sheaf Theory**: Curry, J. (2014). "Sheaves, Cosheaves and Applications"
- **Predictive Coding**: Salvatori et al. (2025). "A survey on neuro-mimetic deep learning via predictive coding"

## Future Work

1. **Nonlinear Extension**: Replace W with Jacobians at equilibria
2. **Real-time Diagnostics**: Efficient online computation of key metrics
3. **Adaptive Initialization**: Use monodromy analysis to guide weight init
4. **Integration with CWD**: Use H¹ reduction as compression progress metric
