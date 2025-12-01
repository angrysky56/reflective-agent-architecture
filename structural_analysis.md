# Reflective Agent Architecture: Structural Analysis

This document provides a structural map of the `reflective-agent-architecture` codebase, breaking down its components, dependencies, and data flow.

## High-Level Architecture

The system is designed as a "Cognitive Workspace" that integrates multiple reasoning frameworks. It operates as an MCP (Model Context Protocol) server, exposing tools to an AI agent.

### Core Systems

1.  **Server ([src/server.py](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/src/server.py))**: The entry point. It initializes the [CognitiveWorkspace](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/src/server.py#124-1618), handles MCP requests, and manages the lifecycle of the application.
2.  **Compass (`src/compass`)**: The central operating system. It orchestrates various "controllers" and "engines" to manage reasoning processes.
3.  **Director (`src/director`)**: Responsible for goal direction, search, and optimization. It ensures the agent stays on track.
4.  **Integration (`src/integration`)**: Connects the core logic to external systems (MCP clients, embeddings) and manages the main reasoning loops.
5.  **Cognition (`src/cognition`)**: Higher-level cognitive functions like "plasticity" and "stereoscopic" reasoning.
6.  **Manifold (`src/manifold`)**: Handles knowledge representation, pattern matching, and embeddings (Hopfield networks, Glove).
7.  **Substrate (`src/substrate`)**: The economic/resource layer. Tracks "energy," costs, and state transitions.

## Directory Map

### `src/`
The root source directory.
-   `server.py`: Main MCP server implementation. Defines `CognitiveWorkspace` and `CWDConfig`.

### `src/compass` (Orchestration & Reasoning)
*   **`compass_framework.py`**: **Core Orchestrator** - Integrates SHAPE, SMART, Executive Controller, SLAP, and Integrated Intelligence. Manages the full task lifecycle: input processing -> objective setting -> meta-cognitive control -> reasoning -> execution -> reflection.
*   **`executive_controller.py`**: **Meta-Cognitive Control** - Coordinates reasoning iterations, manages goals via oMCD, assesses solvability via Self-Discover, and determines when to stop or pivot strategies.
*   **`meta_controller.py`**: **Workflow Adaptation** - Dynamically selects and configures workflows (STANDARD, RESEARCH, CREATIVE, DEBUG) based on task intent analysis.
*   **`integrated_intelligence.py`**: **Decision Synthesis** - Combines multiple intelligence modalities (learning, reasoning, NLU, uncertainty, evolution) using a universal formula. Integrates with LLM providers and the Stereoscopic Engine for intervention gating.
*   **`self_discover_engine.py`**: **Continuous Improvement** - Manages the actor-evaluator-reflection loop. Selects reasoning modules adaptively based on task type and past performance history.
*   **`shape_processor.py`**: **Input Processing** - Handles shorthand expansion, intent extraction, and semantic enrichment of user inputs using LLM analysis and heuristics.
*   **`slap_pipeline.py`**: **Reasoning Pipeline** - Implements the Semantic Logic Auto Progressor (SLAP), generating 8-stage reasoning plans (Conceptualization to Formalization) via LLM.
*   **`smart_planner.py`**: **Objective Management** - Generates and validates SMART (Specific, Measurable, Achievable, Relevant, Time-bound) objectives for tasks.
*   **`omcd_controller.py`**: **Resource Optimization** - Optimizes cognitive resource allocation using MDP, balancing confidence benefits against effort costs.

### `src/director` (Metacognition & Control)
*   **`director_core.py`**: **Director Loop** - The central loop for Phase 1 MVP. Monitors entropy (confusion) from the Processor, detects clashes, computes adaptive beta for exploration, and triggers Manifold search for alternative goal framings.
*   **`entropy_monitor.py`**: **Confusion Detection** - Calculates Shannon entropy from transformer logits to detect "clashes" (high uncertainty). Uses adaptive thresholding based on history.
*   **`matrix_monitor.py`**: **Cognitive Proprioception** - Analyzes the "topology" of thought by downsampling attention matrices into "thumbnails" and projecting them to a "Self-Manifold" to classify states (e.g., Focused, Looping, Scattered).
*   **`sheaf_diagnostics.py`**: **Topological Analysis** - Applies Cellular Sheaf Theory to analyze network consistency. Computes Cohomology (H^0, H^1) to detect irreducible errors and Hodge Decomposition to separate eliminable errors from harmonic residuals. Analyzes Monodromy for feedback loop tension/resonance.
*   **`ltn_refiner.py`**: **Continuous Refinement** - Implements "Operator C" (Belief Revision) as a gradient-based optimization. Generates intermediate waypoints when discrete search fails, satisfying logical constraints (fuzzy logic) and energy barriers.
*   **`hybrid_search.py`**: **Search Strategy** - Combines discrete k-NN search on the Manifold with continuous LTN refinement.
*   **`search_mvp.py`**: **Search Interface** - Defines the `SearchResult` data structure and likely serves as a lightweight interface or base class for search operations.

### `src/integration`
This directory bridges the Cognitive Workspace (CWD) and the Reflective Agent Architecture (RAA), enabling bidirectional communication and control.

-   **`raa_loop.py`**: The main integration loop that orchestrates the `Manifold`, `Processor`, `Pointer`, and `Director`. It implements the "Aha!" loop: generating with the Processor, monitoring entropy with the Director, and triggering reframing (Manifold search) when confusion is detected.
-   **`cwd_raa_bridge.py`**: The central coordinator for CWD-RAA integration. It manages tool library synchronization, monitors entropy of CWD operations (via `EntropyCalculator`), triggers RAA search on confusion, and routes discovered alternatives back to CWD. It also implements "Shadow Monitoring" to simulate cognitive states for the Director.
-   **`continuity_field.py`**: Implements the "Identity Manifold" as a Fiber Bundle (E, B, pi, F). It maintains a temporal trajectory of the agent (Base Space) and state space manifolds (Fiber), ensuring topological coherence and detecting "drift" or "ungroundedness" in interventions.
-   **`sleep_cycle.py`**: Implements the "Night Mode" for offline learning. It performs "Replay" (training the Processor on high-quality episodes) and "Crystallization" (identifying frequent graph patterns in CWD and converting them into reusable Tools).
-   **`agent_factory.py`**: A "System 3" component responsible for spawning specialized, ephemeral agents in response to topological obstructions identified by `SheafDiagnostics`. It creates agents like "Debater" (for Tension Loops), "Explorer" (for H1 Holes), and "Creative" (for Low Overlap).
-   **`reasoning_loop.py`**: Implements a pure embedding-space reasoning loop for tasks like the Remote Associates Test (RAT) or analogical reasoning. It bypasses token generation, working directly with `Manifold` retrieval and `Pointer` updates to find solutions in the latent space.
-   **`embedding_mapper.py`**: Handles the alignment between CWD's graph-based representations and RAA's Hopfield embeddings, currently using a shared `sentence-transformers` model.
-   **`entropy_calculator.py`**: Converts CWD operation results (e.g., hypothesis confidence, synthesis quality) into pseudo-probability distributions and calculates Shannon entropy to provide a "confusion signal" for the Director.
-   **`external_mcp_client.py`**: Manages connections to external MCP servers, aggregating tools from multiple sources.
-   **`precuneus.py`**: (Analyzed previously) Acts as a signal consolidator, merging inputs from various sources (User, CWD, RAA) into a unified context for the agent.
-   **`utility_aware_search.py`**: (Placeholder) Intended to bias RAA's Hopfield energy function with CWD's utility scores, guiding search towards high-utility attractors.
-   **`reinforcement.py`**: (Placeholder) Intended to implement Hebbian reinforcement, strengthening attractors for tools that lead to successful compression in CWD.
-   `external_mcp_client.py`: Connects to other MCP servers.

###### `src/cognition`
This directory implements the "Stereoscopic" cognitive architecture, focusing on dual-layer processing (Base vs. Meta) and validity checking.

-   **`stereoscopic_engine.py`**: The orchestrator of the Dual-Layer Architecture. It integrates the "Unconditioned Condition" (Constraints) and "Ground of Being" (Generative Function) through the `ContinuityField`. It processes interventions by validating them against the `PlasticityGate` and anchoring them in the `ContinuityField`.
-   **`plasticity_gate.py`**: Implements the gating mechanism for structural modification. It decides whether to permit a change based on "Epistemic Uncertainty" and "Identity Preservation" (coherence with the Continuity Field), switching between "Exploration" and "Conservative" modes.
-   **`meta_validator.py`**: Implements the Unified Evaluation Ontology. It evaluates reasoning quality along two orthogonal dimensions: **Coverage** (Completeness) and **Rigor** (Depth/Coherence). It includes logic to reconcile disagreements between these validators (e.g., "Approve Conditional", "Revise for Rigor").
-   **`generative_function.py`**: Represents the active agent or "Ground of Being". It acts as an adapter that converts natural language outputs from the LLM into "Intervention Vectors" (embeddings) that can be processed by the Stereoscopic Engine.utputs.

### `src/manifold`
This directory implements the Associative Memory system using Modern Hopfield Networks.

-   **`hopfield_network.py`**: Implements the continuous Modern Hopfield Network with energy-based retrieval. It features an adaptive beta parameter (inverse temperature) that adjusts attention sharpness based on entropy (confusion) signals.
-   **`pattern_generator.py`**: Enables creative pattern generation beyond simple retrieval. It implements mechanisms for "Conceptual Blending" (linear interpolation), "Pattern Composition" (weighted combination), "Exploratory Perturbation" (noise injection), and "Analogical Mapping" (A:B::C:?).
-   **`pattern_curriculum.py`**: Defines strategies for initializing the Manifold's memory, such as "Random", "Manual", or "Prototype" (clustered) initialization, addressing the "Cold Start" problem.
-   **`patterns.py`**: (Likely) Defines data structures or constants for pattern management.
-   **`glove_loader.py`**: Utility for loading GloVe embeddings (legacy or specific use case).

### `src/substrate`
This directory implements the "Substrate" layer, enforcing physical constraints and energy accounting on cognitive operations.

-   **`substrate_director.py`**: A wrapper around the RAA Director that enforces "Energy-Gated Transitions". It checks the `MeasurementLedger` before allowing operations and records their costs, implementing "Recursive Measurement" (the meta-controller tracks the cost of its own monitoring).
-   **`ledger.py`**: The central accounting system for substrate energy. It tracks the current balance and a history of `MeasurementCost` transactions, raising `InsufficientEnergyError` if an operation exceeds available resources.
-   **`energy_token.py`**: Defines the `EnergyToken` immutable value object, ensuring precise and deterministic arithmetic for energy accounting (using `Decimal`).
-   **`measurement_cost.py`**: Defines the cost structure for operations.
-   **`substrate_quantity.py`**: Likely defines physical quantities or units.
-   **`state_registry.py`**: Manages the registration of system states.
-   **`transition_registry.py`**: Tracks valid state transitions.

### `src/pointer`
This directory implements the "Pointer" or "Goal Controller", responsible for maintaining and evolving the agent's intentionality.

-   **`goal_controller.py`**: The primary Goal Controller implementation using Recurrent Neural Networks (GRU or LSTM). It maintains a persistent goal state that biases the Processor's token generation and can be updated by the Director (e.g., after a "reframing" search).
-   **`state_space_model.py`**: An alternative implementation of the Goal Controller using a simplified State-Space Model (SSM) inspired by S4/Mamba. This offers potentially better long-range dependency modeling than standard RNNs.

### `src/processor`
This directory implements the "Processor", the token-generating engine (System 1) that is biased by the Pointer.

-   **`transformer_decoder.py`**: A custom Transformer Decoder (GPT-style) modified to accept a "Goal State" vector. It exposes its internal attention weights to the Director for topological analysis (e.g., detecting "attention loops"). It outputs both next-token logits and the entropy of the distribution.
-   **`goal_biased_attention.py`**: Implements the specific attention mechanism where the Goal Vector is projected and added as a bias to the attention scores, effectively "steering" the model's focus towards goal-relevant information.

### `src/persistence/`
Data Storage.
-   `work_history.py`: Manages persistence of work and history.

## Key Data Flows

1.  **Request Handling**:
    -   User request -> `server.py` (MCP) -> `CognitiveWorkspace`.
2.  **Reasoning Loop**:
    -   `CognitiveWorkspace` -> `integration/raa_loop.py` -> `compass/compass_framework.py`.
3.  **Decision Making**:
    -   `Compass` consults `Director` (`director/director_core.py`) for guidance.
    -   `Director` uses `hybrid_search.py` and `sheaf_diagnostics.py` to evaluate paths.
4.  **Resource Tracking**:
    -   All operations report to `substrate/ledger.py` to track "energy" usage.
5.  **Memory/Pattern Access**:
    -   `Compass`/`Director` query `manifold/hopfield_network.py` for relevant patterns.

## Dependencies

-   **External**: `neo4j` (Graph DB), `chromadb` (Vector DB), `ollama` (LLM), `numpy`, `torch`.
-   **Internal**: Tightly coupled. `Compass` depends on almost everything. `Director` and `Substrate` are foundational.

