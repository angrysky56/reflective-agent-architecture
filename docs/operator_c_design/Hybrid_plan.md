# Hybrid Operator C Design: LTNs as Topographic Handholds

## Executive Summary

This document specifies a **hybrid belief revision architecture** that integrates:
- **RAA's discrete basin hopping** (macro-level conceptual jumps)
- **Logic Tensor Networks (LTNs)** (micro-level gradient navigation)

**Core Innovation**: LTNs provide "topographic handholds" in steep gradient regions between Hopfield energy basins, enabling smoother tunneling through conceptual barriers.

---

## 1. Architectural Vision

### 1.1 Two-Level Reasoning Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│ LEVEL 1: MACRO-NAVIGATION (RAA)                            │
│ • Hopfield Energy Basins = Conceptual Attractors            │
│ • Discrete jumps between stable concepts                    │
│ • Sheaf cohomology validates structural consistency         │
│ • Director monitors entropy and triggers search             │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   │ Steep Gradient Zone
                   │ (Transition Region)
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ LEVEL 2: MICRO-NAVIGATION (LTN)                            │
│ • Continuous gradient flow within/between basins            │
│ • Fuzzy logic constraints for "near-miss" concepts          │
│ • Provides intermediate waypoints RAA can't see             │
│ • Local optimization without full basin reconstruction      │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Problem Statement

**Current Limitation**: RAA's energy-aware k-NN search can fail in two scenarios:

1. **Sparse Memory**: When Manifold has few stored patterns, k-NN has nothing to retrieve
2. **Steep Gradients**: Transition between distant concepts requires crossing high-energy barriers

**Solution**: Use LTNs to:
- Generate **synthetic waypoints** in sparse regions
- Provide **continuous paths** through steep gradients
- Enable **sub-basin refinement** without full reconstruction

---

## 2. Theoretical Foundation

### 2.1 Complementary Paradigms

| Aspect | RAA (Discrete) | LTN (Continuous) |
|--------|---------------|------------------|
| **State Space** | Energy basins (attractors) | Fuzzy semantic manifold |
| **Transitions** | Basin hopping (discrete jumps) | Gradient flow (smooth deformation) |
| **Search** | k-NN + energy filtering | Projected gradient descent |
| **Consistency** | Sheaf cohomology (topological) | AGM postulates (logical) |
| **Complexity** | O(k) per search | O(iterations × constraints) |
| **When to Use** | Far jumps, conceptual reframing | Near misses, local refinement |

### 2.2 The "Topographic Handhold" Metaphor

Imagine climbing between two mountain peaks (energy basins):

- **RAA**: Teleports you between peaks but can't help with the climb
- **LTN**: Provides handholds on the cliff face for continuous ascent
- **Hybrid**: LTN generates intermediate concepts (handholds), RAA validates them as stable waypoints

### 2.3 Mathematical Formulation

#### RAA Energy Landscape
```
E(x) = -log(Σ exp(βᵢ · sim(x, pᵢ)))  # Hopfield energy
```
- Discrete basins at local minima
- Search via k-NN in {p₁, p₂, ..., pₙ}

#### LTN Loss Function
```
L_total(x') = α·L_dist(x', x) + β·L_ev(x', φ) + γ·L_agm(x')
```
- `L_dist`: Minimal change from current belief x
- `L_ev`: Fit with new evidence φ
- `L_agm`: Consistency with logic constraints

#### Hybrid Integration
```
1. RAA Search: Find nearest basin neighbors via k-NN
2. LTN Refinement: If no valid neighbor found, generate synthetic waypoint
3. Validate: Check if LTN waypoint is energetically reachable
4. Store: Add validated waypoint to Manifold for future use
```

---

## 3. Implementation Architecture

### 3.1 Component Structure

```python
src/
├── director/
│   ├── director_core.py          # Existing: entropy monitoring, k-NN search
│   ├── ltn_refiner.py           # NEW: LTN gradient navigation
│   └── hybrid_search.py         # NEW: Orchestrates RAA + LTN
├── manifold/
│   └── modern_hopfield.py       # Existing: energy evaluation
└── server.py
    └── CognitiveWorkspace.revise()  # NEW: Belief revision interface
```

### 3.2 New Components

#### 3.2.1 LTN Refiner (`src/director/ltn_refiner.py`)

**Purpose**: Generate intermediate waypoints using continuous optimization

```python
class LTNRefiner:
    """
    Local refinement using Logic Tensor Networks.

    Generates synthetic intermediate concepts when k-NN fails.
    """

    def __init__(
        self,
        embedding_dim: int,
        learning_rate: float = 0.01,
        max_iterations: int = 100
    ):
        self.embedding_dim = embedding_dim
        self.lr = learning_rate
        self.max_iters = max_iterations

    def refine(
        self,
        current_belief: torch.Tensor,  # Current basin center
        evidence: torch.Tensor,         # Target concept
        constraints: list[str],         # Natural language rules
        energy_evaluator: callable      # Hopfield energy function
    ) -> Optional[torch.Tensor]:
        """
        Generate intermediate waypoint via gradient descent.

        Returns:
            refined_embedding: Synthetic intermediate concept
            None: If no valid path found
        """
        # Initialize candidate near current belief
        candidate = current_belief.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([candidate], lr=self.lr)

        for iteration in range(self.max_iters):
            optimizer.zero_grad()

            # Loss components
            l_dist = torch.norm(candidate - current_belief)  # Minimal change
            l_ev = 1 - F.cosine_similarity(candidate, evidence)  # Evidence fit
            l_energy = energy_evaluator(candidate)  # Energy penalty

            # Constraint validation (fuzzy)
            l_constraints = self._evaluate_constraints(
                candidate, constraints
            )

            # Total loss (weighted)
            loss = (
                0.2 * l_dist +
                0.5 * l_ev +
                0.2 * l_energy +
                0.1 * l_constraints
            )

            loss.backward()
            optimizer.step()

            # Early stopping if converged
            if loss.item() < 0.1:
                break

        # Validate final candidate
        if self._is_valid_waypoint(candidate, current_belief, energy_evaluator):
            return candidate.detach()
        return None

    def _is_valid_waypoint(
        self,
        candidate: torch.Tensor,
        current: torch.Tensor,
        energy_fn: callable
    ) -> bool:
        """
        Check if candidate is energetically reachable.

        Criteria:
        1. Energy barrier < threshold (reachable)
        2. Not in same basin as current (actually moved)
        3. Stable (local minimum check)
        """
        # Energy barrier check
        energy_diff = abs(energy_fn(candidate) - energy_fn(current))
        if energy_diff > 5.0:  # Unreachable barrier
            return False

        # Basin separation check
        similarity = F.cosine_similarity(candidate, current, dim=0)
        if similarity > 0.95:  # Too similar, didn't escape basin
            return False

        return True
```

#### 3.2.2 Hybrid Search (`src/director/hybrid_search.py`)

**Purpose**: Orchestrate RAA k-NN with LTN fallback

```python
class HybridSearchStrategy:
    """
    Hybrid search combining RAA basin hopping with LTN refinement.

    Strategy:
    1. Try RAA k-NN search (fast, discrete)
    2. If fails (no valid neighbors), use LTN refinement (slow, continuous)
    3. Validate via Sheaf diagnostics
    4. Store successful LTN waypoints in Manifold
    """

    def __init__(
        self,
        manifold,
        ltn_refiner: LTNRefiner,
        sheaf_analyzer: SheafAnalyzer,
        config: DirectorConfig
    ):
        self.manifold = manifold
        self.ltn = ltn_refiner
        self.sheaf = sheaf_analyzer
        self.config = config

    def search(
        self,
        current_state: torch.Tensor,
        evidence: Optional[torch.Tensor] = None,
        constraints: Optional[list[str]] = None,
        context: Optional[dict] = None
    ) -> Optional[SearchResult]:
        """
        Hybrid search with automatic strategy selection.

        Returns:
            SearchResult with source="knn" or source="ltn"
        """
        # Stage 1: Try RAA k-NN (fast path)
        raa_result = self._try_raa_search(current_state)

        if raa_result is not None:
            logger.info("✓ RAA k-NN search succeeded")
            return raa_result

        # Stage 2: Sparse memory or steep gradient → Use LTN
        if evidence is None:
            logger.warning("No evidence for LTN refinement, search failed")
            return None

        logger.info("RAA search failed, attempting LTN refinement...")
        ltn_result = self._try_ltn_refinement(
            current_state, evidence, constraints or []
        )

        return ltn_result

    def _try_raa_search(
        self, current_state: torch.Tensor
    ) -> Optional[SearchResult]:
        """Attempt RAA energy-aware k-NN search."""
        memory = self.manifold.get_patterns()

        if memory.shape[0] < self.config.search_k:
            return None  # Sparse memory

        result = energy_aware_knn_search(
            current_state=current_state,
            memory_patterns=memory,
            energy_evaluator=self.manifold.energy,
            k=self.config.search_k,
            exclude_threshold=self.config.exclude_threshold
        )

        # Validate with Sheaf
        if result and self._sheaf_validates(result.pattern):
            result.source = "knn"
            return result
        return None

    def _try_ltn_refinement(
        self,
        current: torch.Tensor,
        evidence: torch.Tensor,
        constraints: list[str]
    ) -> Optional[SearchResult]:
        """Attempt LTN gradient refinement."""
        waypoint = self.ltn.refine(
            current_belief=current,
            evidence=evidence,
            constraints=constraints,
            energy_evaluator=self.manifold.energy
        )

        if waypoint is None:
            return None

        # Validate with Sheaf
        if not self._sheaf_validates(waypoint):
            return None

        # Store in Manifold for future k-NN
        self.manifold.store(waypoint.unsqueeze(0))

        energy = self.manifold.energy(waypoint)

        return SearchResult(
            pattern=waypoint,
            energy=energy.item(),
            distance=F.cosine_similarity(waypoint, current, dim=0).item(),
            source="ltn"
        )

    def _sheaf_validates(self, pattern: torch.Tensor) -> bool:
        """Check topological consistency via Sheaf cohomology."""
        # Placeholder: requires weight matrix from attention
        # For now, use heuristic based on energy stability
        energy = self.manifold.energy(pattern)
        return energy < 0  # Must be in an attractor
```

---

## 4. Integration with Existing RAA

### 4.1 Minimal Changes Required

**Modified Files:**
1. `src/director/director_core.py`: Replace search method to use HybridSearchStrategy
2. `src/server.py`: Add `revise` tool calling hybrid search

**New Files:**
1. `src/director/ltn_refiner.py`: LTN gradient descent implementation
2. `src/director/hybrid_search.py`: Orchestration layer

### 4.2 Director Core Modifications

```python
# src/director/director_core.py

class DirectorMVP:
    def __init__(self, manifold, config: Optional[DirectorConfig] = None):
        # ... existing initialization ...

        # Add LTN components
        self.ltn_refiner = LTNRefiner(
            embedding_dim=manifold.pattern_dim,
            learning_rate=0.01,
            max_iterations=100
        )

        self.hybrid_search = HybridSearchStrategy(
            manifold=manifold,
            ltn_refiner=self.ltn_refiner,
            sheaf_analyzer=self.sheaf_analyzer,
            config=self.config
        )

    def search(
        self,
        current_state: torch.Tensor,
        evidence: Optional[torch.Tensor] = None,  # NEW parameter
        constraints: Optional[list[str]] = None,   # NEW parameter
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[SearchResult]:
        """Search with hybrid RAA + LTN strategy."""
        return self.hybrid_search.search(
            current_state, evidence, constraints, context
        )
```

### 4.3 Server Interface (Belief Revision)

```python
# src/server.py

class CognitiveWorkspace:
    def revise_belief(
        self,
        node_id: str,
        evidence: str,
        constraints: Optional[list[str]] = None
    ) -> dict:
        """
        Revise a belief node using Hybrid Operator C.

        Workflow:
        1. Retrieve current belief embedding
        2. Embed new evidence
        3. Search with hybrid strategy (RAA → LTN fallback)
        4. Validate with Sheaf diagnostics
        5. Create revision node in graph

        Returns:
            {
                "revision_id": new_node_id,
                "revision_content": generated_text,
                "search_strategy": "knn" | "ltn",
                "energy_cost": float,
                "sheaf_valid": bool
            }
        """
        with self.neo4j_driver.session() as session:
            # 1. Retrieve belief
            result = session.run(
                "MATCH (n:ThoughtNode {id: $id}) RETURN n.content as content",
                id=node_id
            ).single()

            if not result:
                return {"error": f"Node {node_id} not found"}

            belief_text = result["content"]
            belief_embedding = torch.tensor(
                self._embed_text(belief_text), dtype=torch.float32
            )

            # 2. Embed evidence
            evidence_embedding = torch.tensor(
                self._embed_text(evidence), dtype=torch.float32
            )

            # 3. Hybrid search via Director
            search_result = self.bridge.director.search(
                current_state=belief_embedding,
                evidence=evidence_embedding,
                constraints=constraints or []
            )

            if search_result is None:
                return {"error": "No valid revision path found"}

            # 4. Generate revised text from embedding
            # (Use LLM with belief + evidence + search_result as context)
            revised_text = self._generate_revision_text(
                belief_text, evidence, search_result.pattern
            )

            # 5. Store in graph
            new_node_id = self._create_thought_node(
                session,
                revised_text,
                "revision",
                confidence=1.0,
                embedding=search_result.pattern.numpy()
            )

            session.run(
                """
                MATCH (old:ThoughtNode {id: $old_id}),
                      (new:ThoughtNode {id: $new_id})
                CREATE (new)-[:REVISES {
                    strategy: $strategy,
                    energy_cost: $energy
                }]->(old)
                """,
                old_id=node_id,
                new_id=new_node_id,
                strategy=search_result.source,
                energy=search_result.energy
            )

            return {
                "original_id": node_id,
                "revision_id": new_node_id,
                "revision_content": revised_text,
                "evidence": evidence,
                "search_strategy": search_result.source,
                "energy_cost": search_result.energy,
                "sheaf_valid": True
            }
```

---

## 5. Usage Examples

### 5.1 Simple Belief Revision

```python
# User query: "I believed the Earth was 6000 years old, but new evidence
# shows radiometric dating indicates 4.5 billion years"

result = workspace.revise_belief(
    node_id="belief_earth_age",
    evidence="Radiometric dating of oldest rocks shows 4.5 billion years",
    constraints=[
        "Must maintain scientific consistency",
        "Must acknowledge prior belief was incorrect"
    ]
)

# Output:
{
    "revision_content": "While I previously believed the Earth was young,
                        radiometric evidence conclusively shows an age of
                        4.5 billion years, requiring revision of my
                        understanding of geological timescales.",
    "search_strategy": "knn",  # RAA found neighbor in memory
    "energy_cost": 2.3
}
```

### 5.2 Steep Gradient (Requires LTN)

```python
# Query: "I believe free will exists absolutely, but deterministic physics
# suggests all events are causally predetermined"

result = workspace.revise_belief(
    node_id="belief_free_will",
    evidence="Quantum mechanics introduces fundamental indeterminacy",
    constraints=["Must reconcile subjective experience with physics"]
)

# Output:
{
    "revision_content": "Free will may emerge from quantum indeterminacy
                        at neural scales, making determinism incomplete
                        while preserving causal coherence at macro level.",
    "search_strategy": "ltn",  # RAA failed, LTN generated waypoint
    "energy_cost": 7.8  # Higher cost due to steep gradient
}
```

---

## 6. Performance Characteristics

| Scenario | RAA Alone | Hybrid (RAA + LTN) |
|----------|-----------|-------------------|
| **Dense memory, nearby concepts** | ✓ Fast (k-NN) | ✓ Fast (k-NN) |
| **Sparse memory** | ✗ Fails | ✓ LTN generates waypoint |
| **Steep gradient (distant concepts)** | ✗ Fails | ✓ LTN provides path |
| **Complex constraints** | ~ Soft via energy | ✓ Hard via LTN loss |
| **Computational cost** | O(k) | O(k) + O(iters) if LTN needed |

**Key Insight**: Hybrid is strictly superior - it degrades gracefully to LTN only when RAA fails.

---

## 7. Future Enhancements

### 7.1 Meta-Learning LTN Parameters
- Learn optimal `α, β, γ` weights for different domains
- Adapt learning rate based on gradient steepness

### 7.2 Hierarchical LTN
- Multi-scale refinement: coarse → fine waypoints
- Analogous to hierarchical path planning

### 7.3 Constraint Learning
- Learn which constraints are most predictive of valid revisions
- Build library of domain-specific logic rules

### 7.4 Integration with System 3
- When both RAA and LTN fail → escalate to heavy compute (Opus 4, o1)
- Use Sheaf H¹ obstructions as escalation criterion

---

## 8. Implementation Roadmap

**Phase 1: Core LTN Refiner** (1-2 days)
- [ ] Implement `LTNRefiner` class
- [ ] Test gradient descent on toy problems
- [ ] Validate energy constraints work correctly

**Phase 2: Hybrid Search Integration** (1 day)
- [ ] Implement `HybridSearchStrategy`
- [ ] Modify `DirectorMVP.search()` to use hybrid
- [ ] Add unit tests

**Phase 3: Belief Revision Interface** (1 day)
- [ ] Implement `CognitiveWorkspace.revise_belief()`
- [ ] Add MCP tool for `revise`
- [ ] Test with philosophical examples

**Phase 4: Evaluation** (2 days)
- [ ] Test on RAT problems (does LTN improve accuracy?)
- [ ] Test on belief revision benchmarks
- [ ] Compare performance: RAA vs Hybrid vs LTN-only

**Total Estimate**: 5-6 days for MVP

---

## 9. Open Questions

1. **Constraint Encoding**: How to convert natural language constraints to differentiable loss terms?
   - Proposed: Use embedding similarity to constraint statement

2. **Sheaf Integration**: How to apply sheaf cohomology when we don't have attention weights?
   - Proposed: Use graph neighborhood as proxy for weight matrix

3. **LTN Convergence**: What if gradient descent gets stuck in local minimum?
   - Proposed: Multi-start with different initializations

4. **When to Prefer LTN**: Can we predict a priori if LTN will be needed?
   - Proposed: Use memory sparsity + energy barrier heuristics

---

## 10. Conclusion

This hybrid architecture achieves:

✓ **Preserves RAA strengths**: Fast discrete search, topological validation
✓ **Adds LTN capabilities**: Continuous refinement, constraint handling
✓ **Graceful degradation**: LTN only activates when needed
✓ **Minimal integration cost**: ~3 new files, small modifications to existing code
✓ **Theoretically grounded**: Combines discrete (Hopfield) and continuous (LTN) paradigms

**The "topographic handhold" metaphor is realized**: LTNs provide intermediate waypoints that help RAA navigate steep conceptual gradients, enabling smoother tunneling between distant ideas.

**Next Step**: Implement Phase 1 (LTNRefiner) and validate on toy problems before full integration.

