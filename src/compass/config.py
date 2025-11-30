"""
Configuration module for COMPASS framework.

This module centralizes all configurable parameters for the integrated cognitive system,
including oMCD resource allocation, SLAP weights, SMART metrics, and Self-Discover settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class oMCDConfig:
    """Configuration for oMCD (online Metacognitive Control of Decisions) model."""

    # Cost parameters
    alpha: float = 0.1  # Unitary effort cost
    nu: float = 2.0  # Cost power (controls cost variation with resource investment)

    # Effort efficacy parameters
    beta: float = 0.5  # Type #1 effort efficacy (precision improvement)
    gamma: float = 0.3  # Type #2 effort efficacy (value mode perturbation)

    # Confidence parameters
    lambda_conf: float = 1.0  # Confidence scaling factor
    R: float = 10.0  # Importance of making a confident decision

    # Stopping policy
    kappa: float = 1.0  # Effort intensity (resources per unit time)
    threshold_omega: float = 0.5  # Decision threshold

    # Resource limits
    max_resources: float = 100.0  # Maximum cognitive resources available
    min_confidence: float = 0.6  # Minimum acceptable confidence level


@dataclass
class SLAPConfig:
    """Configuration for SLAP (Semantic Logic Auto Progressor) framework."""

    # Advancement formula weights: Advancement = Truth + (alpha * Scrutiny) + (beta * Improvement)
    alpha: float = 0.4  # Weight for scrutiny importance
    beta: float = 0.6  # Weight for improvement importance

    # Validation
    def __post_init__(self):
        if not np.isclose(self.alpha + self.beta, 1.0):
            raise ValueError(f"SLAP weights must sum to 1.0, got {self.alpha + self.beta}")

    # Pipeline stages
    stages: List[str] = field(default_factory=lambda: ["Conceptualization", "Representation", "Facts", "Scrutiny", "Derivation", "RuleBased", "Model", "SemanticFormalization"])

    # MCTS parameters for entity identification
    mcts_iterations: int = 1000
    mcts_exploration_constant: float = 1.414  # sqrt(2)


@dataclass
class SMARTConfig:
    """Configuration for SMART (Strategic Management & Resource Tracking) system."""

    # Objective categories
    objective_categories: List[str] = field(default_factory=lambda: ["decision-making", "ethical-compliance", "learning-capabilities", "performance-optimization"])

    # Metrics configuration
    metrics: Dict[str, str] = field(default_factory=lambda: {"decision-making": "Speed and accuracy metrics", "ethical-compliance": "Compliance level metric", "learning-capabilities": "Improvement rate metric", "performance-optimization": "Efficiency and resource usage metrics"})

    # Timeline defaults (in days)
    default_timeline_days: int = 90

    # Progress monitoring
    progress_check_interval_hours: int = 24
    adjustment_threshold: float = 0.7  # If progress < threshold, trigger adjustment


@dataclass
class SelfDiscoverConfig:
    """Configuration for Self-Discover reinforcement learning framework."""

    # Trial limits
    max_trials: int = 10

    # Memory configuration
    max_memory_size: int = 100  # Maximum number of self-reflections to store

    # Reasoning modules (indices from self_discover_TyMod.txt)
    enabled_reasoning_modules: List[int] = field(default_factory=lambda: list(range(1, 40)))

    # Module selection strategy
    module_selection_strategy: str = "adaptive"  # 'adaptive', 'all', 'random', 'top_k'
    top_k_modules: int = 5  # For 'top_k' strategy

    # Evaluation criteria
    pass_threshold: float = 0.8  # Minimum score to consider task completed


@dataclass
class SHAPEConfig:
    """Configuration for SHAPE (Shorthand Assisted Prompt Engineering) system."""

    # Shorthand dictionary (extensible by users)
    shorthand_dict: Dict[str, str] = field(
        default_factory=lambda: {
            "exp": "experiment",
            "opt": "optimize",
            "eval": "evaluate",
            "impl": "implement",
            "viz": "visualize",
            "dbg": "debug",
            "refac": "refactor",
        }
    )

    # Expansion parameters
    context_window_size: int = 5  # Number of surrounding words for context
    enable_ml_expansion: bool = False  # Use ML for automatic shorthand detection

    # Adaptation parameters
    learning_rate: float = 0.01
    feedback_weight: float = 0.5


@dataclass
class IntegratedIntelligenceConfig:
    """Configuration for Integrated Intelligence multi-modal reasoning."""

    # Universal intelligence weights (for linear and interaction terms)
    linear_weights: Dict[str, float] = field(default_factory=lambda: {"learning": 0.2, "reasoning": 0.25, "nlu": 0.15, "uncertainty": 0.15, "evolution": 0.15, "neural": 0.1})

    # Interaction weights (for pairwise function interactions)
    interaction_weight: float = 0.3

    # Learning parameters
    gamma_discounting: float = 0.95  # Reward discounting factor
    learning_rate: float = 0.001

    # Fuzzy logic parameters
    fuzzy_k: float = 1.0  # Steepness of sigmoid
    fuzzy_c: float = 0.5  # Center point of sigmoid

    # Transfer learning
    delta_learning_factor: float = 0.1

    # Tool usage
    enable_tools: bool = True  # Enable MCP tools for research and information gathering


@dataclass
class ConstraintGovernorConfig:
    """Configuration for Constraint Governor (Reasoning Invariants)."""

    enable_validation: bool = True
    validate_logical_coherence: bool = True
    validate_compositionality: bool = True
    validate_productivity: bool = True
    validate_conceptual_processing: bool = True
    max_contradictions_allowed: int = 0
    coherence_check_depth: int = 3


@dataclass
class ExecutiveControllerConfig:
    """Configuration for Executive Controller (Meta-Cognitive Control)."""

    enable_goal_management: bool = True
    enable_continuous_evaluation: bool = True
    enable_self_awareness: bool = True
    enable_context_awareness: bool = True
    strategy_update_interval: int = 1


@dataclass
class DynamicWorkspaceConfig:
    """Configuration for Dynamic Workspace (Representation Flexibility)."""

    default_representation: str = "sequential"
    allowed_representations: List[str] = field(default_factory=lambda: ["sequential", "hierarchical", "network", "ordinal", "causal", "temporal", "spatial"])
    enable_adaptive_switching: bool = True
    switching_threshold: float = 0.6


@dataclass
class ProceduralToolkitConfig:
    """Configuration for Procedural Toolkit (Reasoning Operations)."""

    enable_backward_chaining: bool = True
    enable_backtracking: bool = True
    enable_restructuring: bool = True
    max_backtrack_depth: int = 5
    operation_timeout_ms: int = 1000


@dataclass
class CGRAConfig:
    """Master configuration for Cognitively-Grounded Reasoning Agent."""

    governor: ConstraintGovernorConfig = field(default_factory=ConstraintGovernorConfig)
    executive: ExecutiveControllerConfig = field(default_factory=ExecutiveControllerConfig)
    workspace: DynamicWorkspaceConfig = field(default_factory=DynamicWorkspaceConfig)
    toolkit: ProceduralToolkitConfig = field(default_factory=ProceduralToolkitConfig)

    # Feature flags for phased rollout
    enable_cgra: bool = False  # Main switch


@dataclass
class COMPASSConfig:
    """Master configuration for the entire COMPASS framework."""

    # Sub-configurations
    omcd: oMCDConfig = field(default_factory=oMCDConfig)
    slap: SLAPConfig = field(default_factory=SLAPConfig)
    smart: SMARTConfig = field(default_factory=SMARTConfig)
    self_discover: SelfDiscoverConfig = field(default_factory=SelfDiscoverConfig)
    shape: SHAPEConfig = field(default_factory=SHAPEConfig)
    intelligence: IntegratedIntelligenceConfig = field(default_factory=IntegratedIntelligenceConfig)
    cgra: CGRAConfig = field(default_factory=CGRAConfig)

    # Global settings
    enable_logging: bool = True
    log_level: str = "INFO"  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    enable_visualization: bool = True

    # Performance settings
    enable_parallel_processing: bool = True
    max_workers: int = 4

    # Persistence
    save_trajectories: bool = True
    save_reflections: bool = True
    output_directory: str = "./compass_output"


# Default configuration instance
DEFAULT_CONFIG = COMPASSConfig()


def get_config() -> COMPASSConfig:
    """Get the default configuration instance."""
    return DEFAULT_CONFIG


def create_custom_config(**kwargs) -> COMPASSConfig:
    """
    Create a custom configuration by overriding default values.

    Args:
        **kwargs: Configuration overrides

    Returns:
        Custom COMPASSConfig instance

    Example:
        >>> config = create_custom_config(
        ...     omcd={'alpha': 0.2, 'R': 15.0},
        ...     slap={'alpha': 0.3, 'beta': 0.7}
        ... )
    """
    config = COMPASSConfig()

    for key, value in kwargs.items():
        if hasattr(config, key):
            if isinstance(value, dict):
                # Update sub-config attributes
                sub_config = getattr(config, key)
                for sub_key, sub_value in value.items():
                    if hasattr(sub_config, sub_key):
                        setattr(sub_config, sub_key, sub_value)
            else:
                setattr(config, key, value)

    return config
