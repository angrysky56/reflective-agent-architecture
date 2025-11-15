"""
Reflective Agent Architecture (RAA)

A research prototype integrating modern associative memory with metacognitive 
monitoring for insight-like problem solving.

Components:
    - Manifold: Modern Hopfield Network for semantic memory
    - Processor: Transformer for sequence generation
    - Pointer: RNN/SSM for goal state management
    - Director: Metacognitive monitor + search engine
"""

__version__ = "0.1.0"
__author__ = "Ty"

from .manifold import Manifold
from .processor import Processor
from .pointer import Pointer
from .director import Director

__all__ = ["Manifold", "Processor", "Pointer", "Director"]
