# RAA Examples

This directory contains example scripts demonstrating the Reflective Agent Architecture.

## Basic Usage

The `basic_usage.py` script demonstrates:

1. Initializing RAA with all four components
2. Storing concept patterns in the Manifold
3. Setting and updating goal states
4. Manifold retrieval via Hopfield dynamics
5. Single-step generation with metacognitive monitoring
6. Full sequence generation with automatic goal reframing
7. Accessing RAA statistics

### Running the example:

```bash
python examples/basic_usage.py
```

## Components Demonstrated

### 1. Manifold (Modern Hopfield Network)
- Pattern storage
- Energy-based retrieval
- Associative memory dynamics

### 2. Processor (Transformer)
- Goal-biased token generation
- Entropy computation
- Autoregressive generation

### 3. Pointer (Goal Controller)
- Goal state maintenance
- Dynamic goal updates
- Persistent state across generation

### 4. Director (Metacognitive Monitor)
- Entropy monitoring
- Clash detection
- k-NN search for alternative framings
- Automatic goal reframing

## The "Aha!" Loop

The complete metacognitive loop:

```
Task Input → Pointer (set goal)
    ↓
Processor (generate with goal bias)
    ↓
Director (monitor entropy)
    ↓
Clash detected? → Search Manifold
    ↓
Update Pointer with new goal
    ↓
Processor (regenerate) → Success!
```

## Next Steps

After running the basic example:

1. Experiment with different entropy thresholds
2. Try different search parameters (k, metric)
3. Store domain-specific concept patterns
4. Evaluate on insight problems (Remote Associates Test, analogies)

See the main README.md for more information about the architecture.
