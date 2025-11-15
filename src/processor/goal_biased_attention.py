"""
Goal-Biased Attention Mechanism

Implements attention that is biased by goal state from the Pointer.
Inspired by Adaptive Transformer Programs (ICLR 2025).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as f


class GoalBiasedAttention(nn.Module):
    """
    Multi-head attention with goal biasing.

    The goal state from Pointer adds an additive bias to attention scores,
    effectively steering the model's focus toward goal-relevant information.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        assert (
            self.head_dim * num_heads == embedding_dim
        ), "embedding_dim must be divisible by num_heads"

        # Standard attention projections
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        # Goal biasing projection
        self.goal_bias_proj = nn.Linear(embedding_dim, num_heads)

        self.dropout = nn.Dropout(dropout)
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        goal_state: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Goal-biased multi-head attention.

        Args:
            query: Query tensor (batch, seq_len, embedding_dim)
            key: Key tensor (batch, seq_len, embedding_dim)
            value: Value tensor (batch, seq_len, embedding_dim)
            goal_state: Goal vector (batch, embedding_dim) or (embedding_dim,)
            attn_mask: Attention mask (seq_len, seq_len)

        Returns:
            output: Attention output (batch, seq_len, embedding_dim)
            attention_weights: Attention weights (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = query.shape

        # Project Q, K, V
        q = self.q_proj(query)  # (batch, seq_len, embedding_dim)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape to multi-head format
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Now: (batch, num_heads, seq_len, head_dim)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # (batch, num_heads, seq_len, seq_len)

        # Apply goal bias if provided
        if goal_state is not None:
            if goal_state.dim() == 1:
                goal_state = goal_state.unsqueeze(0)  # (1, embedding_dim)

            # Project goal to bias per head
            goal_bias = self.goal_bias_proj(goal_state)  # (batch, num_heads)

            # Add bias to attention scores (broadcast across sequence positions)
            goal_bias = goal_bias.unsqueeze(-1).unsqueeze(-1)  # (batch, num_heads, 1, 1)
            scores = scores + goal_bias

        # Apply attention mask
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Softmax to get attention weights
        attention_weights = f.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, other=v)  # (batch, num_heads, seq_len, head_dim)

        # Reshape back
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.embedding_dim)

        # Output projection
        output = self.out_proj(context)

        return output, attention_weights
