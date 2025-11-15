"""
Manifold: Modern Hopfield Network for Associative Memory

Implements the semantic memory component of RAA using Modern Hopfield Networks
with Fenchel-Young energy framework.

Key Papers:
- Hopfield-Fenchel-Young Networks (2024) - arXiv:2411.08590
- Modern Hopfield Networks with Continuous-Time Memories (2025) - arXiv:2502.10122
"""

from .hopfield_network import ModernHopfieldNetwork, HopfieldConfig
from .patterns import PatternMemory

__all__ = ["ModernHopfieldNetwork", "HopfieldConfig", "PatternMemory", "Manifold"]


# Alias for main interface
class Manifold(ModernHopfieldNetwork):
    """
    Main interface for the Manifold component.

    The Manifold stores semantic knowledge as an energy landscape with basin attractors.
    Retrieval happens via associative dynamics that minimize the Hopfield energy function.
    """
    pass
