"""
State-Space Model Goal Controller

Alternative implementation using simplified State-Space Model (S4/Mamba style).
More efficient for long-range dependencies than RNN.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as f


class StateSpaceGoalController(nn.Module):
    """
    Goal controller using State-Space Model.

    Implements a simplified continuous-time state-space model:
        dx/dt = Ax + Bu
        y = Cx + Du

    This provides more efficient long-range dependency modeling than RNNs.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        state_dim: int = 512,
        device: str = "cpu",
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.device = device

        # State-space matrices (learned)
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(state_dim, embedding_dim))
        self.C = nn.Parameter(torch.randn(embedding_dim, state_dim))
        self.D = nn.Parameter(torch.randn(embedding_dim, embedding_dim))

        # State variable
        self.state: Optional[torch.Tensor] = None

    def reset(self, batch_size: int = 1) -> None:
        """Reset state to zero."""
        self.state = torch.zeros(batch_size, self.state_dim).to(self.device)

    def set_goal(self, goal_vector: torch.Tensor) -> None:
        """
        Set goal state directly.

        Args:
            goal_vector: New goal (embedding_dim,) or (batch, embedding_dim)
        """
        if goal_vector.dim() == 1:
            goal_vector = goal_vector.unsqueeze(0)

        batch_size = goal_vector.shape[0]

        if self.state is None:
            self.reset(batch_size)
        assert self.state is not None, "Internal error: state should be initialized after reset()"

        # Update state based on goal
        # x = x + B @ goal
        self.state = self.state + f.linear(goal_vector, self.B.T)
    def get_current_goal(self) -> torch.Tensor:
        """
        Get current goal from state.

        Returns:
            Goal embedding (batch, embedding_dim)
        """
        if self.state is None:
            return torch.zeros(1, self.embedding_dim).to(self.device)
        assert self.state is not None, "Internal error: state should not be None here"

        # y = C @ x
        goal = f.linear(self.state, self.C)

        return goal
        return goal

    def update(self, input_vec: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """
        Update state with input (discrete-time step).

        Args:
            input_vec: Input vector (batch, embedding_dim)
            dt: Time step for discretization

        Returns:
            Output goal
        """
        if input_vec.dim() == 1:
            input_vec = input_vec.unsqueeze(0)

        batch_size = input_vec.shape[0]
        if self.state is None:
            self.reset(batch_size)
        assert self.state is not None, "Internal error: state should be initialized after reset()"

        # Discretize continuous-time system (Euler method)
        # x_{t+1} = x_t + dt * (A @ x_t + B @ u_t)
        state_derivative = f.linear(self.state, self.A.T) + f.linear(input_vec, self.B.T)
        self.state = self.state + dt * state_derivative
        state_derivative = f.linear(self.state, self.A.T) + f.linear(input_vec, self.B.T)
        self.state = self.state + dt * state_derivative

        # Output: y = C @ x + D @ u
        output = f.linear(self.state, self.C) + f.linear(input_vec, self.D)

        return output

    def forward(self, input_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        forward pass.

        Args:
            input_vec: Optional input for state update

        Returns:
            Current goal state
        """
        if input_vec is not None:
            return self.update(input_vec)
        return self.get_current_goal()

    def __repr__(self) -> str:
        return f"StateSpaceGoalController(embedding={self.embedding_dim}, state={self.state_dim})"
