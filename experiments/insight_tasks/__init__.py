"""
RAA Integration Module

Provides integration layers that compose RAA components into functional systems:
- ReasoningLoop: Pure embedding-based reasoning for insight tasks
- GenerationLoop: Token-based generation with metacognitive monitoring (future)
"""

from .reasoning_loop import RAAReasoningLoop, ReasoningConfig

__all__ = [
    "RAAReasoningLoop",
    "ReasoningConfig",
]
