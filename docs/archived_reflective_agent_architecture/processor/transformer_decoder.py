"""
Transformer Decoder Implementation

Standard transformer decoder with goal-biasing capability.
Refactored to expose attention weights for cognitive monitoring.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as f

from .goal_biased_attention import GoalBiasedAttention


@dataclass
class ProcessorConfig:
    """Configuration for Transformer Processor."""

    vocab_size: int = 50257  # GPT-2 vocab size
    embedding_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    feedforward_dim: int = 2048
    max_seq_length: int = 512
    dropout: float = 0.1
    device: str = "cpu"


class CustomDecoderLayer(nn.Module):
    """
    Custom Transformer Decoder Layer that exposes attention weights.
    Uses GoalBiasedAttention for self-attention.
    """

    def __init__(self, config: ProcessorConfig):
        super().__init__()
        self.embedding_dim = config.embedding_dim

        # Self-Attention with Goal Biasing
        self.self_attn = GoalBiasedAttention(
            embedding_dim=config.embedding_dim, num_heads=config.num_heads, dropout=config.dropout
        )

        # Feed-forward network
        self.linear1 = nn.Linear(config.embedding_dim, config.feedforward_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.feedforward_dim, config.embedding_dim)

        # Normalization and Dropout
        self.norm1 = nn.LayerNorm(config.embedding_dim)
        self.norm2 = nn.LayerNorm(config.embedding_dim)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        goal_state: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            output: Layer output (batch, seq_len, embedding_dim)
            attn_weights: Self-attention weights (batch, num_heads, seq_len, seq_len)
        """
        # 1. Self-Attention
        # tgt is (batch, seq_len, embedding_dim)
        tgt2, attn_weights = self.self_attn(
            query=tgt, key=tgt, value=tgt, goal_state=goal_state, attn_mask=tgt_mask
        )

        # Residual connection + Norm
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 2. Feed-Forward
        tgt2 = self.linear2(self.dropout(f.relu(self.linear1(tgt))))

        # Residual connection + Norm
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        return tgt, attn_weights


class TransformerDecoder(nn.Module):
    """
    Transformer decoder for sequence generation.

    Key features:
    - Standard causal self-attention
    - Goal-biasing via additive bias to attention
    - Outputs both tokens and entropy for metacognitive monitoring
    - Exposes attention weights to Director for topological analysis
    """

    def __init__(self, config: ProcessorConfig, director: Optional[Any] = None):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.device = config.device
        self.director = director

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)

        # Positional embedding
        self.position_embedding = nn.Embedding(config.max_seq_length, config.embedding_dim)

        # Custom Transformer Layers
        self.layers = nn.ModuleList([CustomDecoderLayer(config) for _ in range(config.num_layers)])

        # Output projection to vocabulary
        self.output_projection = nn.Linear(config.embedding_dim, config.vocab_size)

        self.dropout = nn.Dropout(config.dropout)

    def _create_causal_mask(self, seq_length: int) -> torch.Tensor:
        """
        Create causal attention mask (prevents attending to future tokens).

        Returns:
            Mask of shape (seq_length, seq_length) with True for masked positions
        """
        # In PyTorch attention, mask usually expects float('-inf') for masked positions
        # or boolean True for positions to IGNORE.
        # GoalBiasedAttention expects True for masked positions (positions to ignore).
        # Causal mask: We want to ignore upper triangle.
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        return mask.to(self.device)

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed input tokens with positional encoding.

        Args:
            input_ids: Token IDs of shape (batch, seq_length)

        Returns:
            Embeddings of shape (batch, seq_length, embedding_dim)
        """
        batch_size, seq_length = input_ids.shape

        # Token embeddings
        token_embeds = self.token_embedding(input_ids)

        # Positional embeddings
        positions = torch.arange(seq_length, device=self.device).unsqueeze(0)
        position_embeds = self.position_embedding(positions)

        # Combine
        embeddings = token_embeds + position_embeds
        embeddings = self.dropout(embeddings)

        return embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        goal_state: Optional[torch.Tensor] = None,
        return_entropy: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        forward pass through transformer.

        Args:
            input_ids: Input token IDs (batch, seq_length)
            goal_state: Optional goal vector from Pointer
            return_entropy: If True, also return entropy of output distribution

        Returns:
            logits: Output logits (batch, seq_length, vocab_size)
            entropy: Optional entropy values (batch, seq_length) if return_entropy=True
        """
        # Embed tokens with positions
        embeddings = self.embed_tokens(input_ids)

        # Create causal mask
        seq_length = input_ids.shape[1]
        causal_mask = self._create_causal_mask(seq_length)

        # Pass through layers
        hidden_states = embeddings
        last_attention_weights = None

        for layer in self.layers:
            hidden_states, attn_weights = layer(
                tgt=hidden_states, goal_state=goal_state, tgt_mask=causal_mask
            )
            last_attention_weights = attn_weights

        # Monitor Thought Process (if Director is attached)
        if self.director is not None and last_attention_weights is not None:
            # We use the attention weights from the last layer
            # Shape: (batch, num_heads, seq_len, seq_len)
            self.director.monitor_thought_process(last_attention_weights.detach())

        # Project to vocabulary
        logits = self.output_projection(hidden_states)

        # Compute entropy if requested
        entropy = None
        if return_entropy:
            probs = f.softmax(logits, dim=-1)
            log_probs = f.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)  # (batch, seq_length)

        return logits, entropy

    def generate_next_token(
        self,
        input_ids: torch.Tensor,
        goal_state: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """
        Generate the next token.

        Returns:
            next_token_id: (batch, 1)
            logits: (batch, vocab_size)
            entropy: scalar entropy of distribution
        """
        logits, entropy_seq = self.forward(input_ids, goal_state=goal_state, return_entropy=True)

        # Get last token logits
        next_token_logits = logits[:, -1, :] / temperature

        # Sample
        probs = f.softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)

        # Get entropy of last step
        entropy = entropy_seq[:, -1].item() if entropy_seq is not None else 0.0

        return next_token_id, next_token_logits, entropy

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        goal_state: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Perform a single training step (Supervised Fine-Tuning).

        Args:
            input_ids: Input tokens (batch, seq_len)
            labels: Target tokens (batch, seq_len) - usually shifted input_ids
            optimizer: PyTorch optimizer
            goal_state: Optional goal vector to condition on

        Returns:
            loss: Scalar loss value
        """
        self.train()
        optimizer.zero_grad()

        logits, _ = self.forward(input_ids, goal_state=goal_state)

        # Flatten for CrossEntropyLoss
        # logits: (batch, seq_len, vocab_size) -> (batch*seq_len, vocab_size)
        # labels: (batch, seq_len) -> (batch*seq_len)
        loss = f.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1))

        loss.backward()
        optimizer.step()

        loss_value: float = loss.item()
        return loss_value

    def __repr__(self) -> str:
        return (
            f"TransformerDecoder(vocab={self.config.vocab_size}, "
            f"dim={self.embedding_dim}, layers={self.config.num_layers})"
        )
