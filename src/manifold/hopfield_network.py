"""
Modern Hopfield Network Implementation

Implements continuous Modern Hopfield Network with energy-based retrieval.
Based on Hopfield-Fenchel-Young framework (arXiv:2411.08590).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HopfieldConfig:
    """Configuration for Modern Hopfield Network."""

    embedding_dim: int = 512
    beta: float = 1.0  # Inverse temperature for softmax
    max_patterns: int = 1000
    update_steps: int = 10  # Number of iterations for convergence
    device: str = "cpu"


class ModernHopfieldNetwork(nn.Module):
    """
    Modern Hopfield Network for associative memory retrieval.

    The energy function is:
        E(ξ) = -lse_β(X^T ξ) + 0.5 ||ξ||²

    where lse_β is the log-sum-exp scaled by β (inverse temperature).

    Retrieval happens via energy minimization through iterative updates:
        ξ_{t+1} = softmax(β X^T ξ_t) X

    This corresponds to the update rule in modern Hopfield networks with
    exponential storage capacity.
    """

    def __init__(self, config: HopfieldConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.beta = config.beta
        self.device = config.device

        # Storage for memory patterns (updated dynamically)
        # Shape: (num_patterns, embedding_dim)
        self.register_buffer("patterns", torch.empty(0, config.embedding_dim))
        self.num_patterns = 0

    def store_pattern(self, pattern: torch.Tensor) -> None:
        """
        Store a new pattern in the Hopfield memory.

        Args:
            pattern: Embedding vector of shape (embedding_dim,) or (batch, embedding_dim)
        """
        if pattern.dim() == 1:
            pattern = pattern.unsqueeze(0)

        # Normalize pattern (important for energy landscape stability)
        pattern = F.normalize(pattern, p=2, dim=-1)

        # Add to pattern storage
        self.patterns = torch.cat([self.patterns, pattern.to(self.device)], dim=0)
        self.num_patterns = self.patterns.shape[0]

    def get_patterns(self) -> torch.Tensor:
        """Return all stored patterns."""
        return self.patterns

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute Hopfield energy for a given state.

        E(ξ) = -lse_β(X^T ξ) + 0.5 ||ξ||²

        Lower energy = more stable attractor.

        Args:
            state: Current state vector (embedding_dim,) or (batch, embedding_dim)

        Returns:
            Energy value(s)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Compute similarity with all patterns
        similarities = torch.matmul(state, self.patterns.T)  # (batch, num_patterns)

        # Log-sum-exp term (scaled by beta)
        lse_term = torch.logsumexp(self.beta * similarities, dim=-1) / self.beta

        # Quadratic regularization
        norm_term = 0.5 * torch.sum(state ** 2, dim=-1)

        # Energy: negative LSE + regularization
        energy = -lse_term + norm_term

        return energy.squeeze()

    def update_step(self, state: torch.Tensor) -> torch.Tensor:
        """
        Single update step in Hopfield dynamics.

        ξ_{t+1} = softmax(β X^T ξ_t) X

        This is the associative retrieval mechanism.

        Args:
            state: Current state (embedding_dim,) or (batch, embedding_dim)

        Returns:
            Updated state after one iteration
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Compute attention weights over stored patterns
        similarities = torch.matmul(state, self.patterns.T)  # (batch, num_patterns)
        attention = F.softmax(self.beta * similarities, dim=-1)  # (batch, num_patterns)

        # Weighted combination of patterns (associative retrieval)
        new_state = torch.matmul(attention, self.patterns)  # (batch, embedding_dim)

        if squeeze_output:
            new_state = new_state.squeeze(0)

        return new_state

    def retrieve(
        self, query: torch.Tensor, num_steps: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve from associative memory via iterative energy minimization.

        Args:
            query: Query vector (embedding_dim,) or (batch, embedding_dim)
            num_steps: Number of update iterations (defaults to config value)

        Returns:
            retrieved_state: Final converged state
            energy_trajectory: Energy at each iteration
        """
        if num_steps is None:
            num_steps = self.config.update_steps

        state = query.clone()
        energies = []

        for _ in range(num_steps):
            energies.append(self.energy(state))
            state = self.update_step(state)

        # Final energy
        energies.append(self.energy(state))
        energy_trajectory = torch.stack(energies)

        return state, energy_trajectory

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: retrieve from memory.

        Args:
            query: Query embedding (embedding_dim,) or (batch, embedding_dim)

        Returns:
            Retrieved state after convergence
        """
        retrieved_state, _ = self.retrieve(query)
        return retrieved_state

    def clear_memory(self) -> None:
        """Clear all stored patterns."""
        self.patterns = torch.empty(0, self.config.embedding_dim).to(self.device)
        self.num_patterns = 0

    def __repr__(self) -> str:
        return (
            f"ModernHopfieldNetwork(patterns={self.num_patterns}, "
            f"dim={self.embedding_dim}, beta={self.beta})"
        )
