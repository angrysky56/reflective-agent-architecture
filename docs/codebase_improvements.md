# Codebase Improvements & Roadmap

Based on a deep structural analysis of the `reflective-agent-architecture`, this document outlines critical implementation gaps, architectural refinements, optimizations, and code health improvements.

## 1. Critical Implementation Gaps (Phase 3 & 4)

These are components explicitly marked as placeholders or `NotImplementedError` that are essential for the full "Reflective" capability.

-   **Utility-Biased Search ([src/integration/utility_aware_search.py](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/src/integration/utility_aware_search.py))**
    -   **Current State**: [compute_biased_energy](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/src/integration/utility_aware_search.py#52-71) raises `NotImplementedError`.
    -   **Requirement**: Implement the energy modification formula $E'(x) = E_{Hopfield}(x) - \lambda \cdot U(x)$.
    -   **Action**: Connect to CWD utility metrics to bias the Hopfield attractor landscape.

-   **Attractor Reinforcement ([src/integration/reinforcement.py](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/src/integration/reinforcement.py))**
    -   **Current State**: [reinforce_from_compression](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/src/integration/reinforcement.py#54-70) raises `NotImplementedError`.
    -   **Requirement**: Implement Hebbian learning updates for patterns that lead to successful CWD compression (tools that "work").
    -   **Action**: Define the update rule for pattern weights based on "Aha!" moments.

-   **Plasticity Gating ([src/cognition/plasticity_gate.py](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/src/cognition/plasticity_gate.py))**
    -   **Current State**: Uses a hardcoded `uncertainty = 0.1` proxy.
    -   **Requirement**: Real epistemic uncertainty estimation (e.g., via ensemble variance or entropy of the Processor's output).
    -   **Action**: Wire the [Processor](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/src/compass/shape_processor.py#15-338)'s entropy signal into the [PlasticityGate](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/src/cognition/plasticity_gate.py#10-93).

## 2. Architectural Refinements

Improvements to the system's design to enhance modularity, scalability, and configurability.

-   **Centralized Embedding Service**
    -   **Issue**: [embedding_mapper.py](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/src/integration/embedding_mapper.py), [generative_function.py](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/src/cognition/generative_function.py), and [hopfield_network.py](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/src/manifold/hopfield_network.py) (indirectly) may instantiate their own embedding models or logic.
    -   **Proposal**: Create a shared `EmbeddingService` or `ModelManager` singleton to load the `sentence-transformers` model once and serve it to all components. This saves VRAM and ensures consistency.

-   **Dynamic Persona Configuration**
    -   **Issue**: [agent_factory.py](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/src/integration/agent_factory.py) likely contains hardcoded system prompts for agents like "Debater" or "Creative".
    -   **Proposal**: Move agent personas to a generic configuration file (YAML/JSON) or a `PromptRegistry`. This allows for easier tuning of agent personalities without code changes.

-   **Sheaf-Theoretic Diagnostics**
    -   **Issue**: [continuity_field.py](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/src/integration/continuity_field.py) currently uses simple distance/coherence metrics. [agent_factory.py](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/src/integration/agent_factory.py) references [SheafDiagnostics](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/src/director/sheaf_diagnostics.py#95-112) but the implementation details in `continuity_field` seem basic.
    -   **Proposal**: Upgrade [ContinuityField](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/src/integration/continuity_field.py#11-166) to explicitly model the "Restriction Maps" and "Cohomology" (obstructions) as described in the theoretical docs.

## 3. Optimizations

Performance and algorithmic improvements.

-   **Hopfield "Cold Start" Strategy**
    -   **Opportunity**: [pattern_curriculum.py](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/src/manifold/pattern_curriculum.py) provides strategies (Prototype, Random), but [raa_loop.py](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/src/integration/raa_loop.py) needs to be verified to ensure it *uses* a smart curriculum on initialization.
    -   **Action**: Ensure the default RAA setup uses [PrototypePatternCurriculum](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/src/manifold/pattern_curriculum.py#113-166) to seed the memory with diverse semantic clusters, preventing the "empty mind" problem.

-   **State-Space Model (SSM) Evaluation**
    -   **Opportunity**: [state_space_model.py](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/src/pointer/state_space_model.py) is an experimental alternative to the RNN Goal Controller.
    -   **Action**: Benchmark SSM vs. RNN for long-context goal stability. If SSM is superior/faster, promote it to default.

## 4. Code Health & Maintenance

-   **Type Safety & Validation**:
    -   Ensure [cwd_raa_bridge.py](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/src/integration/cwd_raa_bridge.py) has robust type checking for the dynamic data flowing between CWD (graph) and RAA (tensor) worlds.
    -   Use `Pydantic` models for the "Intervention Vectors" and "Cognitive States" to enforce schema validity.

-   **Error Handling**:
    -   [external_mcp_client.py](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/src/integration/external_mcp_client.py): Ensure robust timeout and retry logic for network calls to external MCP servers.
    -   [raa_loop.py](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/src/integration/raa_loop.py): Graceful degradation if the [Director](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/src/director/director_core.py#72-568) or `Manifold` fails (e.g., fallback to simple LLM generation).

## 5. TODOs from Code Scans

-   [src/director/search_mvp.py](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/src/director/search_mvp.py): Check for scaling limits or temporary implementations.
-   [src/compass/shape_processor.py](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/src/compass/shape_processor.py): Review specific TODOs related to SHAPE processing.
-   [src/substrate/test_theoretical_validation.py](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/src/substrate/test_theoretical_validation.py): Ensure theoretical tests are passing and cover the "Axioms".

## Roadmap Suggestion

1.  **Phase 3 Implementation**: Focus on [utility_aware_search.py](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/src/integration/utility_aware_search.py) and [plasticity_gate.py](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/src/cognition/plasticity_gate.py) to close the loop between "Thinking" and "Valuing".
2.  **Refactor**: Extract `EmbeddingService`.
3.  **Phase 4 Implementation**: Implement [reinforcement.py](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/src/integration/reinforcement.py) for learning over time.
