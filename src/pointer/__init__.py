"""
Pointer: Goal State Controller

Maintains current goal representation as persistent state.
Implemented as RNN or State-Space Model (S4/Mamba).

The Pointer receives goal updates from the Director after successful search
and provides goal biasing signals to the Processor.
"""

from .goal_controller import GoalController, PointerConfig
from .state_space_model import StateSpaceGoalController

__all__ = ["GoalController", "PointerConfig", "StateSpaceGoalController", "Pointer"]


# Alias for main interface
class Pointer(GoalController):
    """
    Main interface for the Pointer component.

    The Pointer maintains the current goal state and provides biasing
    signals to the Processor component.
    """
    pass
