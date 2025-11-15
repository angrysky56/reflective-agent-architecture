"""
Goal Controller Implementation

Maintains persistent goal state using RNN-based mechanism.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class PointerConfig:
    """Configuration for Goal Controller."""

    embedding_dim: int = 512
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.1
    controller_type: str = "gru"  # "gru", "lstm", or "ssm"
    device: str = "cpu"


class GoalController(nn.Module):
    """
    Goal state controller using RNN.

    The Pointer maintains a persistent goal representation that:
    1. Biases the Processor's token generation
    2. Can be updated by the Director when search finds better framings
    3. Evolves smoothly over time to maintain coherence
    """

    def __init__(self, config: PointerConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.device = config.device

        # RNN for maintaining goal state
        if config.controller_type == "gru":
            self.rnn = nn.GRU(
                input_size=config.embedding_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.num_layers,
                dropout=config.dropout if config.num_layers > 1 else 0,
                batch_first=True,
            )
        elif config.controller_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=config.embedding_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.num_layers,
                dropout=config.dropout if config.num_layers > 1 else 0,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unknown controller type: {config.controller_type}")

        # Project RNN hidden state to goal embedding
        self.goal_projection = nn.Linear(config.hidden_dim, config.embedding_dim)

        # Initialize hidden state
        self.hidden_state = None

    def reset(self, batch_size: int = 1) -> None:
        """Reset the goal controller to initial state."""
        num_layers = self.config.num_layers

        if self.config.controller_type == "lstm":
            # LSTM has both hidden and cell state
            h0 = torch.zeros(num_layers, batch_size, self.hidden_dim).to(self.device)
            c0 = torch.zeros(num_layers, batch_size, self.hidden_dim).to(self.device)
            self.hidden_state = (h0, c0)
        else:
            # GRU has only hidden state
            self.hidden_state = torch.zeros(
                num_layers, batch_size, self.hidden_dim
            ).to(self.device)

    def set_goal(self, goal_vector: torch.Tensor) -> None:
        """
        Directly set the goal state (used by Director after search).

        Args:
            goal_vector: New goal embedding (embedding_dim,) or (batch, embedding_dim)
        """
        if goal_vector.dim() == 1:
            goal_vector = goal_vector.unsqueeze(0).unsqueeze(0)  # (1, 1, embedding_dim)
        elif goal_vector.dim() == 2:
            goal_vector = goal_vector.unsqueeze(1)  # (batch, 1, embedding_dim)

        batch_size = goal_vector.shape[0]

        # Initialize hidden state if needed
        if self.hidden_state is None:
            self.reset(batch_size)

        # Update RNN state with new goal
        _, self.hidden_state = self.rnn(goal_vector, self.hidden_state)

    def get_current_goal(self) -> torch.Tensor:
        """
        Get current goal state for biasing the Processor.

        Returns:
            Goal embedding (batch, embedding_dim)
        """
        if self.hidden_state is None:
            # Return zero goal if not initialized
            return torch.zeros(1, self.embedding_dim).to(self.device)

        # Extract hidden state
        if self.config.controller_type == "lstm":
            hidden = self.hidden_state[0]  # (num_layers, batch, hidden_dim)
        else:
            hidden = self.hidden_state  # (num_layers, batch, hidden_dim)

        # Use last layer's hidden state
        last_hidden = hidden[-1]  # (batch, hidden_dim)

        # Project to goal embedding space
        goal = self.goal_projection(last_hidden)  # (batch, embedding_dim)

        return goal

    def update(self, context: torch.Tensor) -> torch.Tensor:
        """
        Update goal state based on context (gradual evolution).

        Args:
            context: Context embedding (batch, embedding_dim) or (batch, seq_len, embedding_dim)

        Returns:
            Updated goal state
        """
        if context.dim() == 2:
            context = context.unsqueeze(1)  # (batch, 1, embedding_dim)

        batch_size = context.shape[0]

        # Initialize hidden state if needed
        if self.hidden_state is None:
            self.reset(batch_size)

        # RNN forward pass
        _, self.hidden_state = self.rnn(context, self.hidden_state)

        # Get updated goal
        return self.get_current_goal()

    def forward(
        self,
        input_sequence: Optional[torch.Tensor] = None,
        new_goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through goal controller.

        Args:
            input_sequence: Optional input for context-based update
            new_goal: Optional new goal from Director search

        Returns:
            Current goal state
        """
        # Update goal if Director provided new one
        if new_goal is not None:
            self.set_goal(new_goal)

        # Gradual update based on context
        if input_sequence is not None:
            return self.update(input_sequence)

        # Just return current goal
        return self.get_current_goal()

    def __repr__(self) -> str:
        return (
            f"GoalController(type={self.config.controller_type}, "
            f"dim={self.embedding_dim}, hidden={self.hidden_dim})"
        )
