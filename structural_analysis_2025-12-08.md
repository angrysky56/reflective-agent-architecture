# Reflective Agent Architecture: Structural Analysis (2025-12-08)

This document provides a comprehensive structural map of the `reflective-agent-architecture` codebase as of December 8, 2025. It breaks down components, dependencies, and data flow, organized alphabetically by directory for easier navigation.

## Directory Map

### `src/`

The root source directory containing the core MCP server implementation.

- [`server.py`](src/server.py): **Main Entry Point** - Defines `CognitiveWorkspace` MCP Server, initializes all subsystems, and handles tool execution.
- [`raa_history.db`](src/raa_history.db): SQLite database for work history and entropy logs.

### `src/cognition/`

Higher-level "stereoscopic" reasoning and formal logic.

- [`category_theory_engine.py`](src/cognition/category_theory_engine.py): **Categorical Logic** - Treats Knowledge Graph as a formal Category (Objects/Morphisms).
- [`curiosity.py`](src/cognition/curiosity.py): **Intrinsic Motivation** - Drives exploration based on information gaps or novelty.
- [`emotion_framework.py`](src/cognition/emotion_framework.py): **Computational Empathy** - Loader and query interface for the Emotion Evolution Framework.
- [`generative_function.py`](src/cognition/generative_function.py): **Ground of Being** - Converts LLM outputs to intervention vectors.
- [`grok_lang.py`](src/cognition/grok_lang.py): **Empathetic Alignment** - Measures inter-agent alignment (Grok-Depth).
- [`logic_core.py`](src/cognition/logic_core.py): **Formal Verification** - Interface to Prover9/Mace4.
- [`meta_validator.py`](src/cognition/meta_validator.py): **Evaluation** - Unified Ontology (Coverage vs. Rigor).
- [`plasticity_gate.py`](src/cognition/plasticity_gate.py): **Learning Gate** - Controls structural modification based on uncertainty.
- [`stereoscopic_engine.py`](src/cognition/stereoscopic_engine.py): **Dual-Layer Reasoning** - Orchestrates Base vs. Meta layers.
- [`system_guide.py`](src/cognition/system_guide.py): **Architectural Guidance** - High-level system direction.
- [`working_memory.py`](src/cognition/working_memory.py): **Context Management** - Sliding window of recent cognitive operations.

### `src/compass/`

The central operating system for orchestration and reasoning.

- [`adapters.py`](src/compass/adapters.py): **LLM Adaptation** - Adapts RAA's LLM factory for Compass consumption.
- [`compass_framework.py`](src/compass/compass_framework.py): **Core Orchestrator** - Integrates SHAPE, SMART, Executive Controller, SLAP, and Integrated Intelligence.
- [`config.py`](src/compass/config.py): **Configuration** - Settings for Compass modules.
- [`constraint_governor.py`](src/compass/constraint_governor.py): **Constraint Management** - Enforces logical constraints on reasoning.
- [`executive_controller.py`](src/compass/executive_controller.py): **Meta-Cognitive Control** - Coordinates reasoning iterations and asserts solvability.
- [`integrated_intelligence.py`](src/compass/integrated_intelligence.py): **Decision Synthesis** - Combines intelligence modalities using a universal formula.
- [`mcp_tool_adapter.py`](src/compass/mcp_tool_adapter.py): **Tool Bridging** - Exposes Compass functionality as MCP tools.
- [`meta_controller.py`](src/compass/meta_controller.py): **Workflow Adaptation** - Dynamically selects workflows.
- [`omcd_controller.py`](src/compass/omcd_controller.py): **Resource Optimization** - Optimizes cognitive resource allocation using MDP.
- [`orthogonal_dimensions.py`](src/compass/orthogonal_dimensions.py): **Analysis** - Analyzes independence of concepts.
- [`procedural_toolkit.py`](src/compass/procedural_toolkit.py): **Utilities** - Helper functions for procedural generation.
- [`representation_selector.py`](src/compass/representation_selector.py): **Format Selection** - Chooses best output format.
- [`sandbox.py`](src/compass/sandbox.py): **Safe Execution** - Isolated environment for testing code/reasoning.
- [`self_discover_engine.py`](src/compass/self_discover_engine.py): **Continuous Improvement** - Manages the actor-evaluator-reflection loop.
- [`shape_processor.py`](src/compass/shape_processor.py): **Input Processing** - Handles shorthand expansion and intent extraction.
- [`slap_pipeline.py`](src/compass/slap_pipeline.py): **Reasoning Pipeline** - Implements the Semantic Logic Auto Progressor (SLAP).
- [`smart_planner.py`](src/compass/smart_planner.py): **Objective Management** - Generates and validates SMART objectives.
- [`swarm.py`](src/compass/swarm.py): **Swarm Intelligence** - Implements `ConsensusEngine` and `SwarmController`.
- [`system_prompts.py`](src/compass/system_prompts.py): **Prompt Management** - Central repository for system prompts.
- [`utils.py`](src/compass/utils.py): **General Utilities** - Common helper functions.

#### `src/compass/advisors/`
- [`registry.py`](src/compass/advisors/registry.py): **Advisor Management** - Manages profiles for swarm agents.

#### `src/compass/governance/`
- [`amendment.py`](src/compass/governance/amendment.py): **Constitutional Amendments** - Manages changes to governance.
- [`ontology.py`](src/compass/governance/ontology.py): **Ontological Graph** - Manages system ontology.
- [`verification.py`](src/compass/governance/verification.py): **Constitutional Guard** - Implements verification logic.

### `src/config/`

Configuration files.

- [`advisors.example.json`](src/config/advisors.example.json): Template for advisor profiles.
- [`advisors.json`](src/config/advisors.json): Active Advisor profiles.
- [`emotion_evolution_framework.json`](src/config/emotion_evolution_framework.json): Emotion framework definition.
- [`personas.json`](src/config/personas.json): Agent personas.

### `src/dashboard/`

Web interface for visualization.

- [`app.py`](src/dashboard/app.py): **Server** - Flask/Streamlit app implementation.
- [`debug_llm.py`](src/dashboard/debug_llm.py): **Debugging** - LLM debugging utility.
- [`mcp_client_wrapper.py`](src/dashboard/mcp_client_wrapper.py): **Client** - Wrapper for MCP interaction.
- [`style.css`](src/dashboard/style.css): **Styling** - Frontend stylesheets.

#### `src/dashboard/chat_history/`
- Stores JSON logs of chat sessions for playback and analysis.

### `src/director/`

Responsible for metacognition, goal direction, and entropy regulation.

- [`adaptive_criterion.py`](src/director/adaptive_criterion.py): **Dynamic Thresholds** - Manages intervention criteria.
- [`allostatic_controller.py`](src/director/allostatic_controller.py): **Stability Regulation** - Maintains system stability.
- [`director_core.py`](src/director/director_core.py): **Director Loop** - Monitors entropy and triggers reframing.
- [`entropy_monitor.py`](src/director/entropy_monitor.py): **Confusion Detection** - Calculates Shannon entropy.
- [`epistemic_discriminator.py`](src/director/epistemic_discriminator.py): **Knowledge Assessment** - Distinguishes known/unknown.
- [`epistemic_metrics.py`](src/director/epistemic_metrics.py): **Uncertainty Metrics** - Calculates certainty scores.
- [`hybrid_search.py`](src/director/hybrid_search.py): **Search Strategy** - Combines k-NN with LTN refinement.
- [`intervention_tracker.py`](src/director/intervention_tracker.py): **Action Tracking** - Logs interventions.
- [`ltn_refiner.py`](src/director/ltn_refiner.py): **Belief Revision** - Logic Tensor Networks.
- [`matrix_monitor.py`](src/director/matrix_monitor.py): **Cognitive Proprioception** - Classifies thought states.
- [`meta_pattern_analyzer.py`](src/director/meta_pattern_analyzer.py): **Pattern Recognition** - Identifies recurring meta-patterns.
- [`plasticity_modulator.py`](src/director/plasticity_modulator.py): **Learning Control** - Modulates system plasticity.
- [`process_logger.py`](src/director/process_logger.py): **Logging** - Structured logging.
- [`recursive_observer.py`](src/director/recursive_observer.py): **Self-Monitoring** - Higher-order observation.
- [`reflexive_closure_engine.py`](src/director/reflexive_closure_engine.py): **Closure** - Ensures logical completion.
- [`search_mvp.py`](src/director/search_mvp.py): **Prototype** - MVP search logic.
- [`sheaf_diagnostics.py`](src/director/sheaf_diagnostics.py): **Topological Analysis** - Cohomology computation.
- [`simple_gp.py`](src/director/simple_gp.py): **Symbolic Regression** - Formula evolution.
- [`thought_suppression.py`](src/director/thought_suppression.py): **Inhibition** - Suppresses irrelevant thoughts.

### `src/embeddings/`

Vector embedding providers and tools.

- [`base_embedding_provider.py`](src/embeddings/base_embedding_provider.py): **Interface** - Abstract base class.
- [`embedding_factory.py`](src/embeddings/embedding_factory.py): **Factory** - Creates providers.
- [`forensics.py`](src/embeddings/forensics.py): **Diagnostics** - Analysis of embedding distributions.
- [`lmstudio_embedding_provider.py`](src/embeddings/lmstudio_embedding_provider.py): LM Studio integration.
- [`migration_trainer.py`](src/embeddings/migration_trainer.py): **Training** - Trains projection matrices.
- [`ollama_embedding_provider.py`](src/embeddings/ollama_embedding_provider.py): Ollama integration.
- [`openrouter_embedding_provider.py`](src/embeddings/openrouter_embedding_provider.py): OpenRouter integration.
- [`sentence_transformer_provider.py`](src/embeddings/sentence_transformer_provider.py): Local model integration.

#### `src/embeddings/projections/`
- Stores trained projection matrices (`.pt`) for migrating between embedding models.

### `src/integration/`

Bridges Cognitive Workspace (CWD) and RAA.

- [`agent_factory.py`](src/integration/agent_factory.py): **System 3 Spawning** - Creates ephemeral agents.
- [`continuity_field.py`](src/integration/continuity_field.py): **Identity Manifold** - Fiber Bundle implementation.
- [`continuity_service.py`](src/integration/continuity_service.py): **Persistence** - Continuity Field service layer.
- [`cwd_raa_bridge.py`](src/integration/cwd_raa_bridge.py): **Bridge** - Main integration coordinator.
- [`embedding_mapper.py`](src/integration/embedding_mapper.py): **Alignment** - Maps CWD graphs to embeddings.
- [`entropy_calculator.py`](src/integration/entropy_calculator.py): **Signals** - Converts stats to entropy.
- [`external_mcp_client.py`](src/integration/external_mcp_client.py): **Integration** - Connects to external MCPs.
- [`precuneus.py`](src/integration/precuneus.py): **Fusion** - Consolidates context streams.
- [`raa_loop.py`](src/integration/raa_loop.py): **Main Loop** - Orchestrates core components.
- [`reasoning_loop.py`](src/integration/reasoning_loop.py): **Latent Reasoning** - Embedding-space loop.
- [`reinforcement.py`](src/integration/reinforcement.py): **RL** - Reinforcement learning tools.
- [`sleep_cycle.py`](src/integration/sleep_cycle.py): **Offline Learning** - Dreaming and crystallization.
- [`swarm_controller.py`](src/integration/swarm_controller.py): **Hive Mind** - Multi-agent dialectics.
- [`utility_aware_search.py`](src/integration/utility_aware_search.py): **Search** - Utility-guided exploration.

### `src/llm/`

Abstraction layer for Large Language Models.

- [`anthropic_provider.py`](src/llm/anthropic_provider.py): Anthropic integration.
- [`factory.py`](src/llm/factory.py): **Factory** - Instantiates providers.
- [`gemini_provider.py`](src/llm/gemini_provider.py): Google Gemini integration.
- [`huggingface_provider.py`](src/llm/huggingface_provider.py): HuggingFace integration.
- [`ollama_provider.py`](src/llm/ollama_provider.py): Ollama integration.
- [`openai_provider.py`](src/llm/openai_provider.py): OpenAI integration.
- [`openrouter_provider.py`](src/llm/openrouter_provider.py): OpenRouter integration.
- [`provider.py`](src/llm/provider.py): **Interface** - Abstract base class.

### `src/manifold/`

Associative Memory system using Hopfield Networks.

- [`glove_loader.py`](src/manifold/glove_loader.py): **Embeddings** - GloVe loader.
- [`hopfield_network.py`](src/manifold/hopfield_network.py): **Core Memory** - Continuous Hopfield Network.
- [`pattern_curriculum.py`](src/manifold/pattern_curriculum.py): **Initialization** - Cold start logic.
- [`pattern_generator.py`](src/manifold/pattern_generator.py): **Creativity** - Generates new patterns.
- [`patterns.py`](src/manifold/patterns.py): **Storage** - Manages pattern metadata.

### `src/persistence/`

Data persistence.

- [`work_history.py`](src/persistence/work_history.py): **History DB** - Manages SQLite operations for history and entropy.

### `src/pointer/`

Goal Controller and Intentionality.

- [`goal_controller.py`](src/pointer/goal_controller.py): **RNN Controller** - GRU/LSTM based direction.
- [`state_space_model.py`](src/pointer/state_space_model.py): **SSM Controller** - S4/Mamba based direction.

### `src/processor/`

Token Generation (System 1).

- [`goal_biased_attention.py`](src/processor/goal_biased_attention.py): **Bias** - Goal-modulated attention.
- [`transformer_decoder.py`](src/processor/transformer_decoder.py): **Decoder** - Custom transformer.

### `src/scripts/`

Utility and administrative scripts.

- [`force_migration.py`](src/scripts/force_migration.py): **Migration** - CLI tool to force vector DB dimension alignment using projections.

### `src/substrate/`

Economic and Resource Layer (Physics/Economics).

- [`director_integration.py`](src/substrate/director_integration.py): **Gating** - Energy-aware director wrapper.
- [`energy.py`](src/substrate/energy.py): **Logic** - Core energy rules.
- [`energy_token.py`](src/substrate/energy_token.py): **Currency** - Energy units.
- [`entropy.py`](src/substrate/entropy.py): **Disorder** - Entropy units.
- [`ledger.py`](src/substrate/ledger.py): **Accounting** - Central energy ledger.
- [`measurement_cost.py`](src/substrate/measurement_cost.py): **Pricing** - Cost definitions.
- [`state_descriptor.py`](src/substrate/state_descriptor.py): **Metadata** - State descriptions.
- [`state_registry.py`](src/substrate/state_registry.py): **Registry** - System states.
- [`substrate_director.py`](src/substrate/substrate_director.py): **Orchestration** - Substrate-level control.
- [`substrate_quantity.py`](src/substrate/substrate_quantity.py): **Math** - Precise arithmetic.
- [`transition_registry.py`](src/substrate/transition_registry.py): **Flow** - Valid state transitions.

### `src/vectordb_migrate/`

Standalone package for Vector DB Migration.

- [`loss_functions.py`](src/vectordb_migrate/loss_functions.py): **Optimization** - Hybrid loss for alignment (MSE/Cosine/Triplet).
- [`migration.py`](src/vectordb_migrate/migration.py): **Core Logic** - Detector and Migrator classes.

#### `src/vectordb_migrate/integrations/`
- [`chroma_migrator.py`](src/vectordb_migrate/integrations/chroma_migrator.py): **Adapter** - ChromaDB specific migration logic.
