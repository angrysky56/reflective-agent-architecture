# Codebase Improvements & Roadmap (2025-12-10)

**Context:** Updated following the implementation of the **Autonomous Director** and **Sanity Architecture (Contextual Eigen-Sorter)**.
**Source:** Analyzed against `structural_analysis_2025-12-09.md` and TAI/Bio-Digital whitepapers.

## 1. Critical Implementation Gaps (Immediate Priorities)

### Phase 3: The Valuation Loop
The "Thinking" (Director) and "Dreaming" (Sleep Cycle) systems are live. The missing link is "Valuing" — guiding these processes with true utility.

-   **Utility-Biased Search ([src/integration/utility_aware_search.py](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/src/integration/utility_aware_search.py))**
    -   **Problem**: `compute_biased_energy` raises `NotImplementedError`.
    -   **Goal**: Implement $E'(x) = E_{Hopfield}(x) - \lambda \cdot U(x)$ to make the agent "attracted" to high-utility states, not just stable ones.
    -   **Action**: Connect `CuriosityModule` utility scores to the Hopfield energy landscape.

-   **Plasticity Gating ([src/cognition/plasticity_gate.py](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/src/cognition/plasticity_gate.py))**
    -   **Problem**: Uses hardcoded `uncertainty = 0.1` proxy.
    -   **Goal**: Real epistemic uncertainty estimation (e.g., from `EpistemicMetrics`).
    -   **Action**: Wire `EpistemicMetrics.calculate_certainty` into the gate to only allow structural changes when "confident enough".

### Phase 4: The Adaptive Constitution
With the **Eigen-Sorter** (`src/integration/eigen_sorter.py`) in place, we have the mechanism for "Sanity." We need to formalize the rules it checks against.

-   **Diamond Proof Integration**
    -   **Concept**: The current `Drift` metric checks alignment with the *User*. It should also check alignment with *Universal Preconditions* (Logic, Stability, etc.) as per the `Adaptive Constitution` whitepaper.
    -   **Action**: Expand `ContextualEigenSorter.calculate_drift` to include a "Universal Basis" vector derived from the Diamond Proof axioms.

## 2. Architectural Refinements

-   **Centralized Embedding Service**
    -   **Issue**: `embedding_mapper.py`, `generative_function.py`, and `eigen_sorter.py` all potentially load their own embedding models/logic.
    -   **Proposal**: Create a `src/core/model_manager.py` singleton to hold the `sentence-transformers` model and serve vectors to all components. Prevents VRAM bloat.

-   **Cognitive IDE Interface**
    -   **Issue**: The "Sanity" and "Reasoning" happen in the background. The user has no visibility into the "Heatmap of Truth" described in the *Bio-Digital* whitepaper.
    -   **Proposal**: Update `src/dashboard/app.py` to visualize the verify/drift metrics. Show a "Sanity Gauge" (Eigen-Drift) and "Truth Heatmap" (Topology confidence).

## 3. Optimizations

-   **Hopfield Cold Start**
    -   **Action**: Ensure `raa_loop.py` calls `PatternCurriculum` on startup to seed the memory with `Prototype` patterns, avoiding the "empty mind" confusion state.

## 4. Code Health

-   **Type Safety**:
    -   Review `src/integration/eigen_sorter.py` and `sleep_cycle.py` for full type coverage (mypy strict mode).
    -   Ensure `cwd_raa_bridge.py` handles the new `ContextualEigenSorter` data flows robustly.

-   **Testing**:
    -   The verification script `tests/verify_eigen_sorter.py` relies on `src` imports. Ensure `PYTHONPATH` is handled correctly in the repo's `.env` or run scripts.
