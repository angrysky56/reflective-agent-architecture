"""
Manifold: Modern Hopfield Network for Associative Memory

Implements the semantic memory component of RAA using Modern Hopfield Networks
with Fenchel-Young energy framework.

Key Papers:
- Hopfield-Fenchel-Young Networks (2024) - arXiv:2411.08590
- Modern Hopfield Networks with Continuous-Time Memories (2025) - arXiv:2502.10122
"""

from typing import Optional

import torch

from .hopfield_network import HopfieldConfig, ModernHopfieldNetwork
from .pattern_curriculum import (
    ManualPatternCurriculum,
    PatternCurriculum,
    PatternCurriculumConfig,
    PrototypePatternCurriculum,
    RandomPatternCurriculum,
    create_pattern_curriculum,
    initialize_manifold_patterns,
)
from .pattern_generator import (
    PatternGenerator,
    create_novel_pattern_from_neighbors,
)
from .patterns import PatternMemory

__all__ = [
    "ModernHopfieldNetwork",
    "HopfieldConfig",
    "PatternMemory",
    "Manifold",
    "PatternCurriculum",
    "PatternCurriculumConfig",
    "RandomPatternCurriculum",
    "ManualPatternCurriculum",
    "PrototypePatternCurriculum",
    "create_pattern_curriculum",
    "initialize_manifold_patterns",
    "PatternGenerator",
    "create_novel_pattern_from_neighbors",
]


# Alias for main interface
# Main interface for the Manifold component
class Manifold:
    """
    Tripartite Manifold for RAA.

    Splits semantic memory into three specialized tracks:
    1. State (vmPFC): Contexts & Environments (Low Beta)
    2. Agent (amPFC): Personas & Intents (Medium Beta)
    3. Action (dmPFC): Tools & Transitions (High Beta)
    """

    def __init__(self, config: HopfieldConfig):
        self.config = config

        # 1. State Memory (vmPFC) - Broad associations
        state_config = HopfieldConfig(
            embedding_dim=config.embedding_dim,
            beta=config.beta_state,
            device=config.device
        )
        self.state_memory = ModernHopfieldNetwork(state_config)

        # 2. Agent Memory (amPFC) - Intent/Persona
        agent_config = HopfieldConfig(
            embedding_dim=config.embedding_dim,
            beta=config.beta_agent,
            device=config.device
        )
        self.agent_memory = ModernHopfieldNetwork(agent_config)

        # 3. Action Memory (dmPFC) - Precise execution
        action_config = HopfieldConfig(
            embedding_dim=config.embedding_dim,
            beta=config.beta_action,
            device=config.device
        )
        self.action_memory = ModernHopfieldNetwork(action_config)

    def store_pattern(self, pattern, domain: str = "state", metadata: Optional[dict] = None):
        """Store pattern in specific domain."""
        if domain == "state":
            self.state_memory.store_pattern(pattern, metadata)
        elif domain == "agent":
            self.agent_memory.store_pattern(pattern, metadata)
        elif domain == "action":
            self.action_memory.store_pattern(pattern, metadata)
        else:
            # Default to state if unknown
            self.state_memory.store_pattern(pattern, metadata)

    def get_pattern_metadata(self, index: int, domain: str = "state") -> dict:
        """Get metadata for a specific pattern index."""
        if domain == "state":
            return self.state_memory.get_pattern_metadata(index)
        elif domain == "agent":
            return self.agent_memory.get_pattern_metadata(index)
        elif domain == "action":
            return self.action_memory.get_pattern_metadata(index)
        return self.state_memory.get_pattern_metadata(index)

    def retrieve(self, query_dict: dict):
        """
        Retrieve from all three manifolds.

        Args:
            query_dict: {'state': vec, 'agent': vec, 'action': vec}

        Returns:
            Dict with (vector, energy) tuples for each domain.
        """
        # Handle legacy/single-tensor call (default to state memory)
        if isinstance(query_dict, torch.Tensor):
            return self.state_memory.retrieve(query_dict)

        results = {}

        # State
        if 'state' in query_dict:
            vec, energy_traj = self.state_memory.retrieve(query_dict['state'])
            energy = self.state_memory.energy(vec) # Get final energy
            results['state'] = (vec, energy)

        # Agent
        if 'agent' in query_dict:
            vec, energy_traj = self.agent_memory.retrieve(query_dict['agent'])
            energy = self.agent_memory.energy(vec)
            results['agent'] = (vec, energy)

        # Action
        if 'action' in query_dict:
            vec, energy_traj = self.action_memory.retrieve(query_dict['action'])
            energy = self.action_memory.energy(vec)
            results['action'] = (vec, energy)

        return results

    @property
    def hopfield(self):
        """Legacy accessor - returns state memory by default."""
        return self.state_memory

    def get_patterns(self):
        """Delegate to state memory (default domain)."""
        return self.state_memory.get_patterns()

    def energy(self, state):
        """Delegate to state memory."""
        return self.state_memory.energy(state)

    def set_beta(self, beta: float):
        """Set inverse temperature for all memories."""
        self.state_memory.set_beta(beta)
        self.agent_memory.set_beta(beta)
        self.action_memory.set_beta(beta)

    def compute_adaptive_beta(self, entropy: float, max_entropy: Optional[float] = None) -> float:
        """Delegate to state memory for adaptive beta calculation."""
        return self.state_memory.compute_adaptive_beta(entropy, max_entropy)

    @property
    def beta(self) -> float:
        """Return current beta (from state memory)."""
        return self.state_memory.beta

    def get_attention(self, state):
        """Delegate to state memory."""
        return self.state_memory.get_attention(state)

    def to(self, device):
        """Move all sub-networks to device."""
        self.state_memory.to(device)
        self.agent_memory.to(device)
        self.action_memory.to(device)
        return self

    @property
    def num_patterns(self) -> int:
        """Total number of patterns across all domains."""
        return (
            self.state_memory.num_patterns +
            self.agent_memory.num_patterns +
            self.action_memory.num_patterns
        )

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension (assumed same for all)."""
        return self.state_memory.embedding_dim
