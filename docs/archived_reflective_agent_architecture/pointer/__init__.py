"""
Pointer: Goal State Controller

Maintains current goal representation as persistent state.
Implemented as RNN or State-Space Model (S4/Mamba).

The Pointer receives goal updates from the Director after successful search
and provides goal biasing signals to the Processor.
"""

from dataclasses import dataclass

import torch

from .goal_controller import GoalController, PointerConfig
from .state_space_model import StateSpaceGoalController


@dataclass
class PointerState:
    """
    State representation for pointer-based processing.

    Used to track multiple pointers exploring the manifold space.
    """

    positions: torch.Tensor  # (num_pointers, hidden_dim)
    velocities: torch.Tensor  # (num_pointers, hidden_dim)
    attention_weights: torch.Tensor  # (num_pointers, 1) or (num_pointers, seq_len)


__all__ = ["GoalController", "PointerConfig", "StateSpaceGoalController", "Pointer", "PointerState"]


# Alias for main interface
class Pointer(GoalController):
    """
    Main interface for the Pointer component.

    The Pointer maintains the current goal state and provides biasing
    signals to the Processor component.
    """

    pass
