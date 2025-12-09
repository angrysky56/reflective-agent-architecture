# Reflective Agent Architecture: Structural Analysis

This document provides a structural map of the `reflective-agent-architecture` codebase, breaking down its components, dependencies, and data flow.

## Directory Map

### `src/`

The root source directory containing the core MCP server implementation.

- [`server.py`](src/server.py): **Main Entry Point** - Defines `CognitiveWorkspace` MCP Server, initializes all subsystems, and handles tool execution.
- [`raa_history.db`](src/raa_history.db): SQLite database for work history and entropy logs.

### `src/compass/`

The central operating system for orchestration and reasoning.

- [`compass_framework.py`](src/compass/compass_framework.py): **Core Orchestrator** - Integrates SHAPE, SMART, Executive Controller, SLAP, and Integrated Intelligence.
- [`executive_controller.py`](src/compass/executive_controller.py): **Meta-Cognitive Control** - Coordinates reasoning iterations and asserts solvability.
- [`meta_controller.py`](src/compass/meta_controller.py): **Workflow Adaptation** - Dynamically selects workflows (STANDARD, RESEARCH, CREATIVE, DEBUG).
- [`integrated_intelligence.py`](src/compass/integrated_intelligence.py): **Decision Synthesis** - Combines intelligence modalities using a universal formula.
- [`self_discover_engine.py`](src/compass/self_discover_engine.py): **Continuous Improvement** - Manages the actor-evaluator-reflection loop.
- [`shape_processor.py`](src/compass/shape_processor.py): **Input Processing** - Handles shorthand expansion and intent extraction.
- [`slap_pipeline.py`](src/compass/slap_pipeline.py): **Reasoning Pipeline** - Implements the Semantic Logic Auto Progressor (SLAP).
- [`smart_planner.py`](src/compass/smart_planner.py): **Objective Management** - Generates and validates SMART objectives.
- [`omcd_controller.py`](src/compass/omcd_controller.py): **Resource Optimization** - Optimizes cognitive resource allocation using MDP.
- [`swarm.py`](src/compass/swarm.py): **Swarm Intelligence** - Implements `ConsensusEngine` and `SwarmController` for multi-agent hypothesis.
- [`adapters.py`](src/compass/adapters.py): **LLM Adaptation** - Adapts RAA's LLM factory for Compass consumption.
- [`mcp_tool_adapter.py`](src/compass/mcp_tool_adapter.py): **Tool Bridging** - Exposes Compass functionality as MCP tools.
- [`procedural_toolkit.py`](src/compass/procedural_toolkit.py): **Utilities** - Helper functions for procedural generation.
- [`representation_selector.py`](src/compass/representation_selector.py): **Format Selection** - Chooses best output format.
- [`sandbox.py`](src/compass/sandbox.py): **Safe Execution** - Isolated environment for testing code/reasoning.
- [`system_prompts.py`](src/compass/system_prompts.py): **Prompt Management** - Central repository for system prompts.
- [`utils.py`](src/compass/utils.py): **General Utilities** - Common helper functions.

#### `src/compass/advisors/`

- [`registry.py`](src/compass/advisors/registry.py): **Advisor Management** - Manages profiles for swarm agents.

#### `src/compass/governance/`

- [`ontology.py`](src/compass/governance/ontology.py): **Ontological Graph** - Manages system ontology.
- [`verification.py`](src/compass/governance/verification.py): **Constitutional Guard** - Implements verification logic.
- [`amendment.py`](src/compass/governance/amendment.py): **Constitutional Amendments** - Manages changes to governance.

### `src/director/`

Responsible for metacognition, goal direction, and entropy regulation.

- [`director_core.py`](src/director/director_core.py): **Director Loop** - Monitors entropy, detects clashes, and triggers reframing.
- [`entropy_monitor.py`](src/director/entropy_monitor.py): **Confusion Detection** - Calculates Shannon entropy from logits.
- [`matrix_monitor.py`](src/director/matrix_monitor.py): **Cognitive Proprioception** - Analyzes attention topology to classify states.
- [`sheaf_diagnostics.py`](src/director/sheaf_diagnostics.py): **Topological Analysis** - Computes Cohomology to detect irreducible errors.
- [`ltn_refiner.py`](src/director/ltn_refiner.py): **Belief Revision** - Implements Logic Tensor Networks for refinement.
- [`hybrid_search.py`](src/director/hybrid_search.py): **Search Strategy** - Combines k-NN search with LTN refinement.
- [`recursive_observer.py`](src/director/recursive_observer.py): **Self-Monitoring** - Higher-order observation of reasoning.
- [`reflexive_closure_engine.py`](src/director/reflexive_closure_engine.py): **Closure Mechanisms** - Ensures logical closure.
- [`meta_pattern_analyzer.py`](src/director/meta_pattern_analyzer.py): **Pattern Recognition** - Identifies recurring thought patterns.
- [`intervention_tracker.py`](src/director/intervention_tracker.py): **Action Tracking** - Logs and analyzes interventions.
- [`adaptive_criterion.py`](src/director/adaptive_criterion.py): **Dynamic Thresholds** - Manages adaptive intervention criteria.
- [`allostatic_controller.py`](src/director/allostatic_controller.py): **Stability Regulation** - Maintains system stability (allostasis).
- [`epistemic_discriminator.py`](src/director/epistemic_discriminator.py): **Knowledge Assessment** - Distinguishes known/unknown states.
- [`epistemic_metrics.py`](src/director/epistemic_metrics.py): **Uncertainty Metrics** - Calculates epistemic uncertainty.
- [`plasticity_modulator.py`](src/director/plasticity_modulator.py): **Learning Rate Control** - Modulates system plasticity.
- [`process_logger.py`](src/director/process_logger.py): **Logging** - structured logging for Director processes.
- [`search_mvp.py`](src/director/search_mvp.py): **Search Prototype** - MVP implementation of search logic.
- [`simple_gp.py`](src/director/simple_gp.py): **Symbolic Regression** - Genetic programming for formula evolution.
- [`thought_suppression.py`](src/director/thought_suppression.py): **Inhibition** - Mechanisms for suppressing irrelevant thoughts.

### `src/integration/`

Bridges Cognitive Workspace (CWD) and RAA, enabling bidirectional communication.

- [`raa_loop.py`](src/integration/raa_loop.py): **Main Loop** - Orchestrates Manifold, Processor, Pointer, and Director.
- [`cwd_raa_bridge.py`](src/integration/cwd_raa_bridge.py): **Integration Bridge** - Central coordinator for CWD-RAA integration.
- [`continuity_field.py`](src/integration/continuity_field.py): **Identity Manifold** - Fiber Bundle implementation to prevent drift.
- [`continuity_service.py`](src/integration/continuity_service.py): **Persistence Service** - Interact with Continuity Field.
- [`agent_factory.py`](src/integration/agent_factory.py): **System 3 Spawning** - Creates ephemeral agents based on topology.
- [`swarm_controller.py`](src/integration/swarm_controller.py): **Hive Mind** - Orchestrates multi-agent dialectics.
- [`sleep_cycle.py`](src/integration/sleep_cycle.py): **Offline Learning** - Consolidates memories and performs diagrammatic rumination.
- [`embedding_mapper.py`](src/integration/embedding_mapper.py): **Representation Alignment** - Maps CWD graphs to RAA embeddings.
- [`entropy_calculator.py`](src/integration/entropy_calculator.py): **Signal Conversion** - Converts CWD results to entropy signals.
- [`external_mcp_client.py`](src/integration/external_mcp_client.py): **External Tools** - Manages connections to other MCP servers.
- [`precuneus.py`](src/integration/precuneus.py): **Signal Fusion** - Consolidates State/Agent/Action streams into unified context.
- [`reasoning_loop.py`](src/integration/reasoning_loop.py): **Latent Reasoning** - Pure embedding-space reasoning loop.
- [`reinforcement.py`](src/integration/reinforcement.py): **RL Mechanisms** - Reinforcement learning integration.
- [`utility_aware_search.py`](src/integration/utility_aware_search.py): **Targeted Exploration** - Search guided by utility functions.

### `src/cognition/`

Higher-level "stereoscopic" reasoning and formal logic.

- [`stereoscopic_engine.py`](src/cognition/stereoscopic_engine.py): **Dual-Layer Reasoning** - Orchestrates Base vs. Meta layers.
- [`plasticity_gate.py`](src/cognition/plasticity_gate.py): **Learning Gate** - Controls structural modification based on uncertainty.
- [`meta_validator.py`](src/cognition/meta_validator.py): **Evaluation** - Unified Ontology (Coverage vs. Rigor).
- [`generative_function.py`](src/cognition/generative_function.py): **Ground of Being** - Converts LLM outputs to intervention vectors.
- [`curiosity.py`](src/cognition/curiosity.py): **Intrinsic Motivation** - Drives exploration based on gaps/novelty.
- [`system_guide.py`](src/cognition/system_guide.py): **Architectural Guidance** - High-level system direction.
- [`grok_lang.py`](src/cognition/grok_lang.py): **Empathetic Alignment** - Measures inter-agent alignment (Grok-Depth).
- [`working_memory.py`](src/cognition/working_memory.py): **Context Management** - Sliding window of recent cognitive operations.
- [`logic_core.py`](src/cognition/logic_core.py): **Formal Verification** - Interface to Prover9/Mace4.
- [`emotion_framework.py`](src/cognition/emotion_framework.py): **Computational Empathy** - Emotion Evolution Framework integration.
- [`category_theory_engine.py`](src/cognition/category_theory_engine.py): **Categorical Logic** - Treats Knowledge Graph as a formal Category (Objects/Morphisms).

### `src/manifold/`

Associative Memory system using Hopfield Networks.

- [`hopfield_network.py`](src/manifold/hopfield_network.py): **Core Memory** - Continuous Modern Hopfield Network.
- [`patterns.py`](src/manifold/patterns.py): **Pattern Management** - Manages semantic patterns and metadata.
- [`pattern_generator.py`](src/manifold/pattern_generator.py): **Creativity** - Generates patterns via blending/composition.
- [`pattern_curriculum.py`](src/manifold/pattern_curriculum.py): **Initialization** - Cold start strategies.
- [`glove_loader.py`](src/manifold/glove_loader.py): **Embeddings** - GloVe loading utility.

### `src/substrate/`

Economic and Resource Layer (Physics/Economics).

- [`ledger.py`](src/substrate/ledger.py): **Accounting** - Central energy accounting system.
- [`energy.py`](src/substrate/energy.py): **Energy Management** - Core energy logic.
- [`energy_token.py`](src/substrate/energy_token.py): **Currency** - Represents cognitive energy units.
- [`entropy.py`](src/substrate/entropy.py): **Disorder** - Represents system uncertainty.
- [`substrate_quantity.py`](src/substrate/substrate_quantity.py): **Base Value** - Precise decimal arithmetic.
- [`state_descriptor.py`](src/substrate/state_descriptor.py): **State Metadata** - Describes system states.
- [`director_integration.py`](src/substrate/director_integration.py): **Gating** - Wraps Director with energy constraints.
- [`measurement_cost.py`](src/substrate/measurement_cost.py): **Cost Model** - Defines costs for operations.
- [`state_registry.py`](src/substrate/state_registry.py): **Registry** - Manages system states.
- [`transition_registry.py`](src/substrate/transition_registry.py): **Transitions** - Tracks valid state changes.
- [`substrate_director.py`](src/substrate/substrate_director.py): **Orchestration** - Substrate-level direction.

### `src/llm/`

Abstraction layer for Large Language Models.

- [`factory.py`](src/llm/factory.py): **Instantiation** - Creates provider instances.
- [`provider.py`](src/llm/provider.py): **Base Class** - Abstract provider interface.
- [`ollama_provider.py`](src/llm/ollama_provider.py): Ollama integration.
- [`openai_provider.py`](src/llm/openai_provider.py): OpenAI integration.
- [`anthropic_provider.py`](src/llm/anthropic_provider.py): Anthropic integration.
- [`gemini_provider.py`](src/llm/gemini_provider.py): Google Gemini integration.
- [`huggingface_provider.py`](src/llm/huggingface_provider.py): HuggingFace integration.
- [`openrouter_provider.py`](src/llm/openrouter_provider.py): OpenRouter integration.


### `src/persistence/`

Data persistence layer.

- [`work_history.py`](src/persistence/work_history.py): **History Database** - SQLite management for work sessions and entropy logs.

### `src/pointer/`

Goal Controller and Intentionality.

- [`goal_controller.py`](src/pointer/goal_controller.py): **RNN Controller** - GRU/LSTM based goal direction.
- [`state_space_model.py`](src/pointer/state_space_model.py): **SSM Controller** - S4/Mamba based goal direction.

### `src/processor/`

Token Generation (System 1).

- [`transformer_decoder.py`](src/processor/transformer_decoder.py): **Custom Decoder** - Transformer with exposed attention.
- [`goal_biased_attention.py`](src/processor/goal_biased_attention.py): **Biased Attention** - Attention mechanism modulated by Goal Vector.

### `src/embeddings/`

Vector embedding providers.

- [`embedding_factory.py`](src/embeddings/embedding_factory.py): **Factory** - Creates embedding providers.
- [`base_embedding_provider.py`](src/embeddings/base_embedding_provider.py): **Base Class** - Abstract interface.
- [`ollama_embedding_provider.py`](src/embeddings/ollama_embedding_provider.py): Ollama embeddings.
- [`lmstudio_embedding_provider.py`](src/embeddings/lmstudio_embedding_provider.py): LM Studio embeddings.
- [`sentence_transformer_provider.py`](src/embeddings/sentence_transformer_provider.py): Local SentenceTransformers.
- [`openrouter_embedding_provider.py`](src/embeddings/openrouter_embedding_provider.py): OpenRouter embeddings.
- [`forensics.py`](src/embeddings/forensics.py): **Forensics** - Diagnostic tools for embedding distributions and health.
- [`migration_trainer.py`](src/embeddings/migration_trainer.py): **Training** - Trains projection matrices for embedding migration.

### `src/vectordb_migrate/`

Standalone package for Vector DB Migration and Alignment.

- [`migration.py`](src/vectordb_migrate/migration.py): **Core Migration** - Handles reading, projecting, and writing vectors between models.
- [`loss_functions.py`](src/vectordb_migrate/loss_functions.py): **Loss Functions** - Implements `HybridMigrationLoss` (MSE + Cosine + Triplet) for alignment.

### `src/dashboard/`

Web interface for visualization.

- [`app.py`](src/dashboard/app.py): **Server** - Flask/Streamlit app implementation.
- [`mcp_client_wrapper.py`](src/dashboard/mcp_client_wrapper.py): **Client** - Wrapper for MCP interaction.
- [`debug_llm.py`](src/dashboard/debug_llm.py): **Debugging** - LLM debugging utility.

### `src/config/`

Configuration files.

- [`advisors.json`](src/config/advisors.json): Advisor profiles.
- [`personas.json`](src/config/personas.json): Agent personas.
- [`emotion_evolution_framework.json`](src/config/emotion_evolution_framework.json): Emotion framework definition.

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
- **`sleep_cycle.py`**: Implements "Night Mode" for offline learning, crystallization, and **Diagrammatic Rumination** (Category-Theoretic Diagram Chasing).
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
- **`logic_core.py`**: **Formal Verification** - Direct integration with Prover9/Mace4 for First-Order Logic proofs, model finding, and category theory axioms. Self-contained binaries in `ladr/bin`.
- **`emotion_framework.py`**: **Computational Empathy** - Loader and query interface for the Emotion Evolution Framework (`src/config/emotion_evolution_framework.json`). Provides access to basic/complex emotions, evolutionary layers, AI interaction guidelines, empathic templates, and valence-arousal mapping.

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

- **`work_history.py`**: **SQLite History** - Manages persistence of work sessions, operation history, and entropy tracking.
  - **Entropy Logging**: Each operation logs its entropy for trend analysis.
  - **Search**: Tokenized multi-word search with OR logic.
  - **Methods**: `log_operation()`, `get_recent_history()`, `get_entropy_history()`, `search_history()`.

### Database Architecture

- **Neo4j (Graph)**: Stores `ThoughtNode` structure and relationships (SYNTHESIZES_FROM, HYPOTHESIZES_CONNECTION_TO, etc.).
- **ChromaDB (Vector)**: Stores document content and embeddings. Path: `chroma_data/` at project root (41MB of persistent data).
- **SQLite (History)**: Stores operation logs, entropy, and metabolic transactions. Path: `src/raa_history.db`.

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
