# Reflective Agent Architecture: Structural Analysis

This document provides a structural map of the `reflective-agent-architecture` codebase, breaking down its components, dependencies, and data flow.

## High-Level Architecture

The system is designed as a "Cognitive Workspace" that integrates multiple reasoning frameworks. It operates as an MCP (Model Context Protocol) server, exposing tools to an AI agent.

### Core Systems

1.  **Server ([src/server.py](src/server.py))**: The entry point. It initializes the `CognitiveWorkspace`, handles MCP requests, and manages the lifecycle of the application.
2.  **Compass (`src/compass`)**: The central operating system. It orchestrates various "controllers" and "engines" to manage reasoning processes, including the new Swarm Intelligence modules.
3.  **Director (`src/director`)**: Responsible for goal direction, search, and optimization. It ensures the agent stays on track using entropy monitoring and topological analysis.
4.  **Integration (`src/integration`)**: Connects the core logic to external systems (MCP clients, embeddings) and manages the main reasoning loops (RAA Loop, Continuity Field).
5.  **Cognition (`src/cognition`)**: Higher-level cognitive functions like "plasticity" and "stereoscopic" reasoning.
6.  **Manifold (`src/manifold`)**: Handles knowledge representation, pattern matching, and embeddings (Hopfield networks, Pattern Memory).
7.  **Substrate (`src/substrate`)**: The economic/resource layer. Tracks "energy," costs, and state transitions using precise accounting.
8.  **LLM (`src/llm`)**: Abstraction layer for Large Language Model providers (Ollama, OpenAI, Anthropic, etc.).

## Directory Map

### `src/`

The root source directory.

- `server.py`: Main MCP server implementation. Defines `CognitiveWorkspace` and `CWDConfig`.

### `src/compass` (Orchestration & Reasoning)

- **`compass_framework.py`**: **Core Orchestrator** - Integrates SHAPE, SMART, Executive Controller, SLAP, and Integrated Intelligence. Manages the full task lifecycle.
- **`executive_controller.py`**: **Meta-Cognitive Control** - Coordinates reasoning iterations, manages goals via oMCD, and assesses solvability.
- **`meta_controller.py`**: **Workflow Adaptation** - Dynamically selects and configures workflows (STANDARD, RESEARCH, CREATIVE, DEBUG).
- **`integrated_intelligence.py`**: **Decision Synthesis** - Combines multiple intelligence modalities using a universal formula.
- **`self_discover_engine.py`**: **Continuous Improvement** - Manages the actor-evaluator-reflection loop.
- **`shape_processor.py`**: **Input Processing** - Handles shorthand expansion and intent extraction.
- **`slap_pipeline.py`**: **Reasoning Pipeline** - Implements the Semantic Logic Auto Progressor (SLAP).
- **`smart_planner.py`**: **Objective Management** - Generates and validates SMART objectives.
- **`omcd_controller.py`**: **Resource Optimization** - Optimizes cognitive resource allocation using MDP.
- **`swarm.py`**: **Swarm Intelligence** - Implements `ConsensusEngine` (Maynard-Cross Learning) and `SwarmController` for multi-agent hypothesis generation and aggregation.
- **`adapters.py`**: **LLM Adaptation** - Adapts RAA's LLM factory for Compass consumption.
- **`mcp_tool_adapter.py`**: **Tool Bridging** - Exposes Compass functionality as MCP tools.
- **`procedural_toolkit.py`**: **Utilities** - Helper functions for procedural generation and text processing.
- **`representation_selector.py`**: **Format Selection** - Chooses the best output format (Markdown, JSON, etc.).
- **`sandbox.py`**: **Safe Execution** - Isolated environment for testing code or reasoning steps.
- **`system_prompts.py`**: **Prompt Management** - Central repository for system prompts.
- **`utils.py`**: **General Utilities** - Common helper functions.
- **`advisors/`**:
  - `registry.py`: **Advisor Management** - Manages `AdvisorProfile`s for swarm agents (e.g., "Linearist", "Periodicist").
- **`governance/`**: **Meta-System Governance**
  - `ontology.py`: Manages the Ontological Graph.
  - `verification.py`: Implements `ConstitutionalGuard`.
  - `amendment.py`: Manages constitutional amendments.

### `src/director` (Metacognition & Control)

- **`director_core.py`**: **Director Loop** - Monitors entropy, detects clashes, and triggers reframing.
- **`entropy_monitor.py`**: **Confusion Detection** - Calculates Shannon entropy from transformer logits.
- **`matrix_monitor.py`**: **Cognitive Proprioception** - Analyzes attention topology to classify states (Focused, Looping, etc.).
- **`sheaf_diagnostics.py`**: **Topological Analysis** - Computes Cohomology (H^0, H^1) to detect irreducible errors.
- **`ltn_refiner.py`**: **Belief Revision** - Implements "Operator C" (Logic Tensor Networks) for gradient-based refinement.
- **`hybrid_search.py`**: **Search Strategy** - Combines discrete k-NN search with continuous LTN refinement.
- **`recursive_observer.py`**: **Self-Monitoring** - Implements higher-order observation of the reasoning process itself.
- **`reflexive_closure_engine.py`**: **Closure Mechanisms** - Ensures reasoning processes reach logical closure.
- **`meta_pattern_analyzer.py`**: **Pattern Recognition** - Identifies recurring meta-patterns in thought processes.
- **`intervention_tracker.py`**: **Action Tracking** - Logs and analyzes interventions for effectiveness.

### `src/integration`

Bridges CWD and RAA, enabling bidirectional communication.

- **`raa_loop.py`**: The main integration loop orchestrating Manifold, Processor, Pointer, and Director.
- **`cwd_raa_bridge.py`**: Central coordinator for CWD-RAA integration.
- **`continuity_field.py`**: Implements the "Identity Manifold" as a Fiber Bundle to prevent drift.
- **`continuity_service.py`**: Service layer for interacting with the Continuity Field.
- **`agent_factory.py`**: **System 3** - Spawns specialized ephemeral agents based on topological needs.
- **`swarm_controller.py`**: **Swarm Intelligence** - Orchestrates "Hive Mind" dynamics with multiple advisors running in parallel and synthesizing via Hegelian dialectics.
- **`sleep_cycle.py`**: Implements "Night Mode" for offline learning and crystallization.
- **`embedding_mapper.py`**: Aligns CWD graph representations with RAA embeddings.
- **`entropy_calculator.py`**: Converts CWD results into entropy signals.
- **`external_mcp_client.py`**: Manages connections to external MCP servers.
- **`precuneus.py`**: Signal consolidator merging inputs into a unified context.
- **`reasoning_loop.py`**: Pure embedding-space reasoning loop.

### `src/cognition`

Implements "Stereoscopic" cognitive architecture.

- **`stereoscopic_engine.py`**: Orchestrator of Dual-Layer Architecture (Base vs. Meta).
- **`plasticity_gate.py`**: Gating mechanism for structural modification based on epistemic uncertainty.
- **`meta_validator.py`**: Unified Evaluation Ontology (Coverage vs. Rigor).
- **`generative_function.py`**: "Ground of Being" - Converts LLM outputs to intervention vectors.
- **`curiosity.py`**: **Intrinsic Motivation** - Drives exploration based on information gaps or novelty.
- **`system_guide.py`**: **Guidance System** - Provides high-level architectural guidance.
- **`grok_lang.py`**: **Empathetic Alignment** - Implements Grok-Depth scoring across six cognitive levels (Signal, Symbol, Syntax, Semantics, Pragmatics, Meta) for measuring inter-agent alignment.
- **`working_memory.py`**: **Short-term Context** - Maintains a sliding window of recent cognitive operations (deconstruct, synthesize, hypothesize, etc.) for LLM continuity. Injects context into all LLM calls for coherent multi-step reasoning.

### `src/manifold`

Associative Memory system.

- **`hopfield_network.py`**: Continuous Modern Hopfield Network with energy-based retrieval.
- **`patterns.py`**: **Pattern Memory** - Manages semantic patterns, labels, and metadata.
- **`pattern_generator.py`**: Generates creative patterns via blending and composition.
- **`pattern_curriculum.py`**: Strategies for initializing memory (Cold Start).
- **`glove_loader.py`**: Utility for loading GloVe embeddings.

### `src/substrate`

Economic and Resource Layer.

- **`ledger.py`**: Central accounting system for energy.
- **`energy.py`**: **Energy Token** - Represents cognitive energy units.
- **`entropy.py`**: **Entropy Token** - Represents system disorder/uncertainty.
- **`substrate_quantity.py`**: **Base Value Object** - precise decimal arithmetic for physical quantities.
- **`state_descriptor.py`**: **State Metadata** - Dataclass for describing system states (energy level, description).
- **`director_integration.py`**: **Substrate Director** - Wraps Director with energy-gating logic.
- **`measurement_cost.py`**: Defines cost structures for operations.
- **`state_registry.py`**: Manages system state registration.
- **`transition_registry.py`**: Tracks valid state transitions.

### `src/llm`

Large Language Model Abstraction Layer.

- **`factory.py`**: **LLM Factory** - Creates provider instances based on configuration.
- **`provider.py`**: **Base Provider** - Abstract base class for LLM providers.
- **`ollama_provider.py`**: Ollama integration.
- **`openai_provider.py`**: OpenAI integration.
- **`anthropic_provider.py`**: Anthropic integration.
- **`gemini_provider.py`**: Google Gemini integration.
- **`huggingface_provider.py`**: HuggingFace integration.
- **`openrouter_provider.py`**: OpenRouter integration.

### `src/pointer`

Goal Controller / Intentionality.

- **`goal_controller.py`**: RNN-based Goal Controller (GRU/LSTM).
- **`state_space_model.py`**: SSM-based Goal Controller (S4/Mamba).

### `src/processor`

Token Generation (System 1).

- **`transformer_decoder.py`**: Custom Transformer Decoder with exposed attention weights.
- **`goal_biased_attention.py`**: Attention mechanism biased by the Goal Vector.

### `src/persistence`

Data Storage.

- **`work_history.py`**: Manages persistence of work sessions and history.

### `experiments/`

Empirical Validation Suite (Diamond Proof).

- **`stats_utils.py`**: Statistical testing suite.
- **`config.py`**: Experiment configuration.
- **`entropy_reduction_test.py`**: Exp 1 (Entropy).
- **`ess_stability_sim.py`**: Exp 2 (ESS).
- **`reflexivity_test.py`**: Exp 3 (Reflexivity).
- **`adversarial_probing.py`**: Exp 4 (Non-Harm).
- **`cantorian_limits_test.py`**: Exp 5 (Cantorian Limits).
- **`info_theory_test.py`**: Exp 6 (Compression).
- **`network_robustness_sim.py`**: Exp 7 (Robustness).
- **`gradient_of_intelligence.py`**: Exp 8 (Gradient).

### `Topological_Active_Inference/`

Theoretical foundations for Topological Active Inference (TAI) and Theory of Mind (ToM) integration.

- **`category_theory_proof.md`**: Formal proof that ToM embeds faithfully in TAI (but not isomorphism). False beliefs → β₀ fractures, Hidden emotions → β₁ cycles.
- **`topological_tomography_analysis.md`**: Methodology for discovering "unknown unknowns" via discordance pattern reconstruction (Conceptual Splatting → Multiple Angles → Inverse Problem Solution).
- **`white_paper.md`**: Core TAI theory.
- **`tom_synthesis.md`**, **`theoretical_assessment.md`**: Supporting analysis.

### `formal_proofs/`

Formal Verification (Prover9/Mace4).

- `T5_ESS.in`, `T6_DirectorCorrectness.in`, `T7_Thermodynamics.in`, `T8_InformationTheory.in`, `T9_SystemsBiology.in`.

## Key Data Flows

1.  **Request Handling**: User request -> `server.py` -> `CognitiveWorkspace`.
2.  **Reasoning Loop**: `CognitiveWorkspace` -> `integration/raa_loop.py` -> `compass/compass_framework.py`.
3.  **Decision Making**: `Compass` -> `Director` (via `substrate/director_integration.py` for energy gating).
4.  **Swarm Consensus**: `Compass` -> `swarm.py` (ConsensusEngine) -> `advisors/registry.py`.
5.  **Synthesis Auto-Resolution**: `synthesize` -> Critique Classification -> If ACTIONABLE: `Director.process_task_with_time_gate()` -> COMPASS.
6.  **Resource Tracking**: All operations -> `substrate/ledger.py`.
7.  **Memory Access**: `Compass`/`Director` -> `manifold/hopfield_network.py` & `manifold/patterns.py`.

## Dependencies

- **External**: `neo4j`, `chromadb`, `ollama`, `numpy`, `torch`, `scipy`, `pandas`.
- **Internal**: Highly coupled architecture. `Compass` is the orchestrator, `Director` provides metacognition, `Substrate` provides physics/economics, and `Manifold` provides memory.
