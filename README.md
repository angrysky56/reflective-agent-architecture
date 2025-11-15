# Reflective Agent Architecture (RAA)

**A Novel Cognitive Architecture for Metacognitive AI**

## Overview

The Reflective Agent Architecture (RAA) is a research prototype that integrates modern associative memory, metacognitive monitoring, and dynamic goal reframing to enable AI systems to detect confusion, search for alternative conceptual framings, and achieve insight-like problem solving.

## Theoretical Foundations

RAA synthesizes three active research frontiers:

1. **Modern Hopfield Networks**: Exponential-capacity associative memory with connections to transformer attention (2024 Nobel Prize in Physics)
2. **Entropy-Based Metacognition**: Detection of model uncertainty through predictive entropy monitoring
3. **Uncertainty-Aware Processing**: Dynamic attention mechanisms that adapt to varying levels of uncertainty

## Core Components

### 1. The Manifold (Associative Memory)
- **Implementation**: Modern Hopfield Network with Fenchel-Young energy framework
- **Function**: Stores semantic knowledge as energy landscape with basin attractors
- **Key Papers**:
  - Hopfield-Fenchel-Young Networks (2024)
  - Modern Hopfield Networks with Continuous-Time Memories (2025)

### 2. The Processor (Transformer)
- **Implementation**: Standard transformer decoder
- **Function**: Token-level sequence generation biased by current goal state
- **Integration**: Receives biasing signals from Pointer component

### 3. The Pointer (Goal Controller)
- **Implementation**: RNN or State-Space Model (S4/Mamba)
- **Function**: Maintains current goal representation as persistent state
- **Update Mechanism**: Receives new goal vectors from Director after successful search

### 4. The Director (Metacognitive Monitor + Search Engine)
- **Monitor**: Shannon entropy calculation on Transformer output distribution
- **Function**: Detects "clashes" (high-entropy states) and triggers search
- **Search Engine**: Structured exploration of Manifold to find alternative framings
- **Key Innovation**: Entropy-triggered associative search for goal reframing

## The "Aha!" Loop

```
1. Task Input → Pointer sets initial goal
2. Processor generates response (biased by goal)
3. Director monitors entropy → Detects "clash"
4. Director suppresses current goal
5. Director searches Manifold for alternative basin
6. Director updates Pointer with new goal
7. Processor resumes with new framing → Success
```

## Implementation Plan

### Segments 1-2: Minimal Viable Architecture
- [ ] Implement Modern Hopfield Network (Manifold)
- [ ] Implement Transformer decoder (Processor)
- [ ] Implement entropy monitor (Director - Part 1)
- [ ] Test: Can system detect confusion?

### Segments 3-4: Director Prototype
- [ ] Implement search mechanism in Hopfield space
- [ ] Implement goal update mechanism (Pointer)
- [ ] Test: Does retrieval reduce entropy?

### Segments 5-6: Integration & Benchmarking
- [ ] Full loop integration
- [ ] Evaluate on insight problems (Remote Associates Test, analogies)
- [ ] Compare to baseline transformer

## Research Questions

1. Can the Director learn when to search vs when to persist?
2. What's the optimal Manifold representation (discrete vs continuous)?
3. Which search strategy is most effective (gradient-based, k-NN, noise injection)?
4. How many search hops before declaring failure?
5. Can latent-space reasoning outperform token-based chain-of-thought?

## Novel Contributions

1. **Explicit metacognitive loop** in neural architecture
2. **Entropy-triggered search** in associative memory
3. **Latent-space reasoning** vs token-generation paradigm
4. **Computational theory of insight** bridging connectionism and symbolism

## Dependencies

Core libraries (to be added to pyproject.toml):
- PyTorch >= 2.0
- transformers (HuggingFace)
- scipy (entropy calculations)
- numpy
- Optional: PyTorch Geometric (if using hybrid GNN approach)

## Project Structure

```
reflective-agent-architecture/
├── src/
│   ├── manifold/       # Modern Hopfield Network implementation
│   ├── processor/      # Transformer components
│   ├── pointer/        # Goal controller
│   ├── director/       # Metacognitive monitor + search
│   └── integration/    # Full RAA loop
├── tests/              # Unit and integration tests
├── docs/               # Detailed documentation and theory
├── experiments/        # Benchmark tasks and evaluation
└── notebooks/          # Jupyter notebooks for exploration
```

## References

See `docs/REFERENCES.md` for complete bibliography of theoretical foundations.

## License

MIT License (to be added)

## Authors

Ty (Primary Researcher)
Development initiated: November 2025
