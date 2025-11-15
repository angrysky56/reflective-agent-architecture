# RAA Integration Architecture

**Author**: Ty  
**Date**: 2025-11-15  
**Status**: Design Document

## Executive Summary

This document specifies the integration layer that composes the four RAA components (Manifold, Processor, Pointer, Director) into functional reasoning and generation systems.

## The Integration Challenge

### Component Abstraction Levels

| Component | Input Space | Output Space | State Space |
|-----------|-------------|--------------|-------------|
| **Manifold** | Embeddings | Embeddings | Patterns (embeddings) |
| **Processor** | Token IDs | Logits | None (stateless) |
| **Pointer** | Embeddings | Embeddings | Goal state (embedding) |
| **Director** | Embeddings + Logits | Embeddings | Entropy history |

### The Gap

**No clear handoff between:**
1. Token generation (Processor) ↔ Embedding reasoning (Manifold/Director)
2. Discrete outputs (token IDs) ↔ Continuous representations (embeddings)
3. Sequential generation ↔ Parallel search

## Design: Two Integration Modes

### Mode 1: Generation Loop (Token-Based)

**Use Case**: Language modeling, text generation, QA

**Flow**:
```
Input tokens → Embed → [Goal biasing] → Generate logits → Monitor entropy
                ↑                                              ↓
                └──── Director search ← Clash detected? ──────┘
                      Returns new goal
```

**API**:
```python
class RAAGenerationLoop:
    def generate(
        self, 
        input_ids: torch.Tensor,
        max_length: int = 100
    ) -> tuple[torch.Tensor, dict]:
        """Generate tokens with metacognitive monitoring."""
```

### Mode 2: Reasoning Loop (Embedding-Based)

**Use Case**: RAT, analogies, conceptual reasoning (no text output needed)

**Flow**:
```
Input embeddings → Goal state → Manifold retrieval → Energy evaluation
                    ↑                                        ↓
                    └──── Director search ← High energy? ───┘
```

**API**:
```python
class RAAReasoningLoop:
    def reason(
        self,
        input_embeddings: torch.Tensor,
        max_steps: int = 10
    ) -> tuple[torch.Tensor, dict]:
        """Pure embedding-based reasoning."""
```

## Key Architectural Decisions

### Decision 1: Two Loops vs Unified Loop

**Chosen**: Two separate loops

**Rationale**:
- Generation and reasoning have fundamentally different requirements
- Clarity over abstraction (explicit is better than implicit)
- Easier to optimize each mode independently
- Avoids feature flags and conditional complexity

### Decision 2: Pseudo-Logits for Reasoning Mode

**Problem**: Director expects logits, but reasoning mode has no vocabulary

**Solution**: Use pattern attention distribution as "pseudo-logits"

**Rationale**:
- Entropy over pattern distribution is semantically meaningful
- Aligns with Hopfield attention mechanism
- No artificial vocabulary mapping needed
- Director's entropy threshold still applicable

### Decision 3: Pointer Integration

**Both loops use Pointer for goal state management**

**Rationale**:
- Consistent goal state evolution
- Can handle both discrete updates (Director search) and gradual evolution
- Provides temporal smoothing
- Enables future policy learning (which goals work)

## Implementation Plan

### Phase 1: Core Integration (Current)
- [x] Component interfaces documented
- [ ] RAAReasoningLoop implementation (priority for RAT)
- [ ] RAAGenerationLoop implementation
- [ ] Unit tests for both loops

**Status**: Design complete, ready for implementation
