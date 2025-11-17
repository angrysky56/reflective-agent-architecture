"""
RAA Integration Module

This module contains:
1. Core RAA integration (RAAReasoningLoop) - Manifold + Director + Pointer
2. CWD-RAA integration (Bridge, Mapper, etc.) - RAA + CWD coordination

Core RAA Components (for embedding-based reasoning):
- RAAReasoningLoop: Full reasoning loop for insight tasks
- ReasoningConfig: Configuration for reasoning loop

CWD-RAA Integration Components (for confusion-triggered utility-guided reframing):
- CWDRAABridge: Main orchestrator
- EmbeddingMapper: Convert between CWD and RAA spaces
- EntropyCalculator: Convert CWD operations to entropy signals
- UtilityAwareSearch: Bias RAA search with CWD utility (Phase 3)
- AttractorReinforcement: Update Manifold from CWD compression (Phase 4)

Usage:

    # Core RAA reasoning
    from src.integration import RAAReasoningLoop, ReasoningConfig

    loop = RAAReasoningLoop(manifold, director, pointer)
    solution, metrics = loop.reason(input_embeddings)

    # CWD-RAA integration
    from src.integration import CWDRAABridge, EmbeddingMapper

    bridge = CWDRAABridge(cwd_server, raa_director, manifold)
    result = bridge.execute_monitored_operation('hypothesize', params)
"""

# Core RAA integration
from .cwd_raa_bridge import BridgeConfig, CWDRAABridge
from .embedding_mapper import EmbeddingMapper
from .entropy_calculator import EntropyCalculator, cwd_to_logits
from .raa_loop import RAAConfig, ReflectiveAgentArchitecture
from .reasoning_loop import RAAReasoningLoop, ReasoningConfig
from .reinforcement import AttractorReinforcement
from .utility_aware_search import UtilityAwareSearch, utility_biased_energy

__all__ = [
    # Core RAA (full architecture with processor/pointer)
    "RAAConfig",
    "ReflectiveAgentArchitecture",
    # Core RAA (embedding-based reasoning loop)
    "RAAReasoningLoop",
    "ReasoningConfig",
    # CWD-RAA Bridge
    "CWDRAABridge",
    "BridgeConfig",
    "EmbeddingMapper",
    "EntropyCalculator",
    "cwd_to_logits",
    "UtilityAwareSearch",
    "utility_biased_energy",
    "AttractorReinforcement",
]

__version__ = "0.1.0"
