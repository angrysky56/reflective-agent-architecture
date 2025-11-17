# RAA ⟷ CWD Integration Design

## Executive Summary

This document outlines the technical architecture for integrating **Reflective Agent Architecture (RAA)** with **Cognitive Workspace Database (CWD)** to create a unified "confusion-triggered utility-guided reframing" system.

**Core Thesis**: RAA's entropy-based confusion detection + CWD's utility-guided exploration = A complete model of how minds transform confusion into mastery.

---

## 1. Conceptual Foundation

### RAA Strengths
- **Entropy monitoring**: Detects stuck/confused states
- **Frame-shifting**: Searches for alternative conceptual framings
- **Hopfield memory**: Energy landscape for semantic patterns

### CWD Strengths
- **Utility filtering**: Prevents "junk food curiosity"
- **Compression progress**: Measures learning improvement
- **Topology tunneling**: Discovers deep analogies

### Integration Hypothesis

RAA's Director can **monitor CWD's System 2 reasoning** operations:
1. CWD generates thought-nodes during complex reasoning
2. RAA monitors entropy of synthesis/hypothesis operations
3. High entropy → RAA triggers analogical search
4. RAA finds alternative framing from CWD's tool library
5. CWD uses new framing for topology tunneling
6. Success → compress to tool → update RAA Manifold

---

## 2. Technical Architecture

### 2.1 Data Flow

```
┌─────────────────────────────────────────────────────────┐
│                    Unified System                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐         ┌──────────────┐             │
│  │  CWD System 2│         │ RAA Director │             │
│  │              │         │              │             │
│  │ • deconstruct│◄────────┤• Entropy Mon │             │
│  │ • hypothesize│         │• Search      │             │
│  │ • synthesize │────────►│• Manifold    │             │
│  │ • constrain  │         │              │             │
│  └──────────────┘         └──────────────┘             │
│         │                        │                      │
│         │                        │                      │
│  ┌──────▼────────┐        ┌─────▼──────┐              │
│  │  Tool Library │◄───────┤  Manifold  │              │
│  │  (compressed) │        │ (attractors)│              │
│  └───────────────┘        └────────────┘              │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Integration Points

#### Point 1: Tool Library → Manifold Storage

**Purpose**: Store CWD's compressed tools as Hopfield attractors

**Implementation**:
```python
# In CWD: After compress_to_tool() succeeds
tool_embedding = get_tool_embedding(tool_id)  # Get compressed representation
raa_manifold.store_pattern(tool_embedding)     # Store as attractor

# Optionally: Strengthen attractor based on utility
compression_strength = compute_compression_score(tool_id)
utility_weight = get_utility_score(tool_id)
attractor_strength = compression_strength * utility_weight
```

**Data Structure**:
```python
@dataclass
class ToolAttractor:
    tool_id: str                    # CWD tool identifier
    embedding: torch.Tensor          # Pattern in Manifold
    utility_score: float            # From CWD goals
    compression_score: float        # Learning improvement
    usage_count: int                # Reinforcement metric
```

#### Point 2: Entropy Spike → Topology Tunneling

**Purpose**: Use RAA's confusion signal to trigger CWD's analogical search

**Implementation**:
```python
def integrated_reasoning_step(
    cwd_operation: str,  # 'hypothesize', 'synthesize', etc.
    node_ids: list[str],
    context: dict,
) -> dict:
    """
    Execute CWD operation with RAA monitoring.
    """
    # Execute CWD operation
    result = cwd_execute(cwd_operation, node_ids)
    
    # Generate logits/distribution for entropy calculation
    logits = cwd_to_logits(result)  # Convert CWD result to probability dist
    
    # RAA monitors entropy
    is_clash, entropy_value = raa_director.check_entropy(logits)
    
    if is_clash:
        # High entropy = confusion/stuck
        # RAA searches for alternative framing
        current_framing = get_current_framing_embedding(node_ids)
        alternative = raa_director.check_and_search(
            current_state=current_framing,
            processor_logits=logits,
            context=context
        )
        
        if alternative is not None:
            # Map alternative back to CWD tool
            tool_id = manifold_to_tool_id(alternative)
            
            # Use tool for topology tunneling
            tunnel_result = cwd_hypothesize_with_tool(
                node_a_id=node_ids[0],
                node_b_id=node_ids[1],
                guiding_tool=tool_id
            )
            
            return tunnel_result
    
    return result
```

#### Point 3: Utility Score → Search Bias

**Purpose**: Guide RAA's Hopfield search toward high-utility regions

**Implementation**:
```python
def utility_biased_energy(state: torch.Tensor, pattern: torch.Tensor) -> float:
    """
    Compute energy with utility bias.
    
    E_utility(ξ) = E_hopfield(ξ) - λ * U(tool)
    
    Where U(tool) is the utility score of the tool represented by the pattern.
    """
    # Standard Hopfield energy
    hopfield_energy = manifold.energy(state)
    
    # Get utility score for this pattern
    tool_id = pattern_to_tool_id(pattern)
    utility_score = cwd_get_utility_score(tool_id)
    
    # Combine (λ is bias weight)    utility_bias = 0.3  # λ parameter
    biased_energy = hopfield_energy - utility_bias * utility_score
    
    return biased_energy

# Update search to use biased energy
def utility_aware_knn_search(
    current_state: torch.Tensor,
    memory_patterns: torch.Tensor,
    cwd_tool_mapping: dict,  # pattern_idx → tool_id
    k: int = 5
) -> SearchResult:
    """
    k-NN search with utility bias.
    """
    energies = []
    for i, pattern in enumerate(memory_patterns):
        energy = utility_biased_energy(current_state, pattern)
        energies.append(energy)
    
    # Select k patterns with lowest biased energy
    top_k_indices = torch.topk(-torch.tensor(energies), k).indices
    
    # Standard k-NN from here...
```

#### Point 4: Compression Progress → Attractor Strength

**Purpose**: Reinforce successful tools in Manifold based on learning improvement

**Implementation**:
```python
def update_manifold_from_compression(
    tool_id: str,
    compression_improvement: float
):
    """
    Strengthen Hopfield attractor when tool proves useful.
    
    Compression improvement indicates learning occurred,
    so we want to make this tool easier to retrieve in future.
    """
    # Get pattern embedding
    pattern_idx = tool_to_pattern_idx(tool_id)
    pattern = manifold.get_patterns()[pattern_idx]
    
    # Compute reinforcement weight
    # Higher compression → stronger pattern
    reinforcement = 1.0 + (compression_improvement * 0.5)
    
    # Scale pattern (increases its basin of attraction)
    reinforced_pattern = pattern * reinforcement
    
    # Update in manifold
    manifold.update_pattern(pattern_idx, reinforced_pattern)
    
    # Alternative: Duplicate pattern (Hebbian reinforcement)
    # "Neurons that fire together wire together"
    manifold.store_pattern(pattern.clone())
```

---

## 3. Implementation Phases

### Phase 1: Infrastructure (Weeks 1-2)

**Goal**: Establish bidirectional communication

**Tasks**:
1. Create `cwd_raa_bridge.py` module
2. Implement embedding conversion:
   - `cwd_node_to_embedding()`: Graph node → vector
   - `embedding_to_cwd_query()`: Vector → CWD search query
3. Create tool-pattern mapping:
   - `ToolManifoldMapper` class
   - Persistent storage (SQLite or JSON)
4. Implement `cwd_to_logits()` for entropy calculation

**Deliverables**:
- [ ] Bridge module with conversion functions
- [ ] Tool-pattern bidirectional mapping
- [ ] Unit tests for conversions

### Phase 2: Entropy-Triggered Search (Weeks 3-4)

**Goal**: RAA monitors CWD operations

**Tasks**:
1. Wrap CWD operations with entropy monitoring
2. Implement `integrated_reasoning_step()`
3. Add entropy tracking to CWD operations:
   - `hypothesize()` → measure hypothesis quality distribution
   - `synthesize()` → measure synthesis confidence
   - `constrain()` → measure constraint satisfaction
4. Create `EntropyBasedTrigger` policy

**Deliverables**:
- [ ] Monitored CWD operation wrapper
- [ ] Entropy calculation from CWD results
- [ ] Trigger policy with threshold tuning

### Phase 3: Utility-Biased Search (Weeks 5-6)

**Goal**: CWD utility guides RAA search

**Tasks**:
1. Modify RAA's energy function with utility bias
2. Implement `utility_biased_energy()`
3. Create `get_active_goal_utilities()` in CWD
4. Tune λ (utility weight) parameter
5. Add utility logging to search episodes

**Deliverables**:
- [ ] Utility-aware energy computation
- [ ] Parameter tuning experiments
- [ ] Performance comparison vs unbiased search

### Phase 4: Bidirectional Learning (Weeks 7-8)

**Goal**: Success in one system reinforces the other

**Tasks**:
1. Implement attractor strengthening from compression
2. Add `update_manifold_from_compression()`
3. Create feedback loop:
   - RAA finds tool → CWD uses it → compression improves → RAA strengthens attractor
4. Implement attractor decay (forget unused tools)
5. Add meta-learning metrics

**Deliverables**:
- [ ] Compression-based reinforcement
- [ ] Attractor strength dynamics
- [ ] Meta-learning evaluation

### Phase 5: Evaluation & Optimization (Weeks 9-10)

**Goal**: Validate integration benefits

**Tasks**:
1. Design integrated benchmark tasks
2. Compare three conditions:
   - RAA alone
   - CWD alone
   - Integrated system
3. Measure:
   - Solution quality
   - Time to solution
   - Tool reuse rate
   - Entropy reduction dynamics
4. Optimize hyperparameters
5. Write integration paper

**Deliverables**:
- [ ] Benchmark results
- [ ] Performance analysis
- [ ] Integration whitepaper

---

## 4. Technical Challenges & Solutions

### Challenge 1: Embedding Space Alignment

**Problem**: CWD uses graph + vector space, RAA uses Hopfield embedding space. How to align?

**Solutions**:
1. **Shared Embedding Model**: Both use same base model (e.g., sentence-transformers)
2. **Projection Layer**: Learn linear map between spaces
3. **Contrastive Learning**: Train alignment via successful tool uses

**Recommendation**: Start with Solution 1 (simplest), upgrade to 2 if needed.

### Challenge 2: Entropy Calculation from CWD

**Problem**: CWD operations don't naturally produce probability distributions.

**Solutions**:
1. **Confidence Scores**: Convert similarity scores to pseudo-probabilities
2. **Ensemble Uncertainty**: Run operation multiple times, measure variance
3. **Learned Predictor**: Train model to predict entropy from operation context

**Recommendation**: Solution 1 for MVP, Solution 3 for production.

### Challenge 3: Utility-Energy Trade-off

**Problem**: Pure utility bias might ignore Hopfield's natural energy landscape.

**Solutions**:
1. **Weighted Combination**: `E_total = α*E_hopfield + (1-α)*(-U)`
2. **Constrained Optimization**: Minimize energy subject to utility > threshold
3. **Pareto Optimization**: Multi-objective search

**Recommendation**: Solution 1 with learned α.

### Challenge 4: Scalability

**Problem**: Large tool libraries → large Manifolds → slow search.

**Solutions**:
1. **Hierarchical Manifolds**: Multiple levels of abstraction
2. **Active Forgetting**: Prune low-utility attractors
3. **Approximate Search**: Use FAISS for k-NN acceleration

**Recommendation**: Combination of 2 (forget) + 3 (fast search).

---

## 5. Expected Benefits

### For RAA
1. **Richer Memory**: Tools are more structured than raw embeddings
2. **Goal-Directed Search**: Utility bias prevents irrelevant frame-shifts
3. **Meta-Learning**: Compression feedback improves future searches

### For CWD
1. **Confusion Detection**: Entropy signal catches stuck states early
2. **Alternative Framings**: RAA provides novel analogical jumps
3. **Exploration Control**: Adaptive beta balances breadth vs depth

### For Unified System
1. **Complete Learning Loop**: Confusion → Search → Insight → Compression
2. **Waste Reduction**: Utility filtering + entropy triggering
3. **Emergent Metacognition**: System learns what confuses it and how to escape

---

## 6. Research Questions

### Empirical
1. Does entropy-triggered topology tunneling outperform always-on analogical search?
2. What is optimal utility bias λ for different task types?
3. How does attractor reinforcement affect long-term performance?

### Theoretical
4. Can we prove convergence of the integrated system?
5. What is the Kolmogorov complexity of problems solved vs unsolved?
6. Does the system exhibit meta-level compression progress?

### Philosophical
7. Is this architecture a computational model of insight?
8. What's the relationship between entropy (RAA) and compression progress (CWD)?
9. Can utility-guided confusion resolution explain human problem-solving?

---

## 7. Code Structure

### New Modules
```
reflective-agent-architecture/
├── src/
│   ├── integration/              # NEW
│   │   ├── __init__.py
│   │   ├── cwd_raa_bridge.py    # Core integration
│   │   ├── embedding_mapper.py   # Space alignment
│   │   ├── entropy_calculator.py # CWD → logits
│   │   ├── utility_aware_search.py # Biased energy
│   │   └── reinforcement.py      # Attractor updates
│   ├── director/
│   │   ├── director_core.py     # MODIFIED: add utility bias
│   │   └── ...
│   └── manifold/
│       ├── hopfield_network.py   # MODIFIED: add update_pattern()
│       └── ...
└── tests/
    └── test_integration.py       # NEW
```

### Key Classes

```python
class CWDRAABridge:
    """Main integration orchestrator."""
    def __init__(self, cwd_server, raa_director, manifold)
    def execute_monitored_operation(self, op, params)
    def sync_tools_to_manifold(self)
    def handle_entropy_spike(self, context)

class EmbeddingMapper:
    """Align CWD and RAA embedding spaces."""
    def cwd_node_to_vector(self, node_id) -> torch.Tensor
    def vector_to_cwd_query(self, vec) -> str
    def learn_alignment(self, pairs)

class UtilityAwareSearch:
    """RAA search with CWD utility bias."""
    def biased_energy(self, state, pattern, utility)
    def search_with_utility(self, state, utilities)

class AttractorReinforcement:
    """Update Manifold from CWD compression."""
    def reinforce_from_compression(self, tool_id, improvement)
    def decay_unused(self, threshold)
```

---

## 8. Success Metrics

### Integration Quality
- [ ] Zero-latency overhead for non-confused states
- [ ] <100ms for entropy check + search trigger
- [ ] Tool library ↔ Manifold sync in <1s

### Performance Improvements
- [ ] 20%+ faster solution time on complex problems
- [ ] 30%+ reduction in dead-end explorations  
- [ ] 50%+ increase in tool reuse rate

### Emergent Properties
- [ ] System discovers meta-strategies (tools about tools)
- [ ] Entropy decreases over time (learning to not get stuck)
- [ ] Utility-alignment improves (fewer junk food curiosities)

---

## 9. Next Immediate Steps

### For Ty to Decide
1. **Scope**: Start with Phase 1 (infrastructure) or jump to Phase 2 (entropy monitoring)?
2. **Codebase**: Modify RAA repo or create new integrated repo?
3. **Testing**: Real tasks or synthetic benchmarks first?

### Technical Prep
1. Ensure CWD server exports needed functions
2. Check RAA Manifold's `update_pattern()` capability
3. Design `cwd_to_logits()` conversion strategy

### Documentation
1. Add integration examples to README
2. Create tutorial notebook
3. Document API changes

---

**Status**: 🎯 Design complete, ready for implementation
**Estimated Effort**: 10 weeks for full integration
**Risk Level**: Medium (novel architecture but well-motivated)
**Potential Impact**: High (complete model of confusion → mastery)