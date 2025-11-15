"""
Transformer Decoder Implementation

Standard transformer decoder with goal-biasing capability.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as f


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


class TransformerDecoder(nn.Module):
    """
    Transformer decoder for sequence generation.

    Key features:
    - Standard causal self-attention
    - Goal-biasing via additive bias to attention
    - Outputs both tokens and entropy for metacognitive monitoring
    """

    def __init__(self, config: ProcessorConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.device = config.device

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)

        # Positional embedding
        self.position_embedding = nn.Embedding(config.max_seq_length, config.embedding_dim)

        # Transformer layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_layers,
        )

        # Output projection to vocabulary
        self.output_projection = nn.Linear(config.embedding_dim, config.vocab_size)

        # Goal biasing mechanism
        self.goal_projection = nn.Linear(config.embedding_dim, config.embedding_dim)

        self.dropout = nn.Dropout(config.dropout)

    def _create_causal_mask(self, seq_length: int) -> torch.Tensor:
        """
        Create causal attention mask (prevents attending to future tokens).

        Returns:
            Mask of shape (seq_length, seq_length) with True for masked positions
        """
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

    def apply_goal_bias(
        self,
        embeddings: torch.Tensor,
        goal_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply goal biasing to embeddings.

        Args:
            embeddings: Token embeddings (batch, seq_length, embedding_dim)
            goal_state: Goal vector from Pointer (embedding_dim,) or (batch, embedding_dim)

        Returns:
            Biased embeddings
        """
        if goal_state is None:
            return embeddings

        # Project goal to same space
        if goal_state.dim() == 1:
            goal_state = goal_state.unsqueeze(0)  # (1, embedding_dim)

        goal_bias = self.goal_projection(goal_state)  # (batch, embedding_dim)

        # Add goal bias to all positions (broadcast)
        biased_embeddings = embeddings + goal_bias.unsqueeze(1)

        return biased_embeddings

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

        # Apply goal biasing
        embeddings = self.apply_goal_bias(embeddings, goal_state)

        # Create causal mask
        seq_length = input_ids.shape[1]
        causal_mask = self._create_causal_mask(seq_length)

        # Transformer forward pass
        # Note: for decoder-only, we use tgt=memory=embeddings
        hidden_states = self.transformer(
            tgt=embeddings,
            memory=embeddings,
            tgt_mask=causal_mask,
            memory_mask=causal_mask,
        )

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
        Generate next token (used for autoregressive generation).

        Args:
            input_ids: Input sequence (batch, seq_length)
            goal_state: Optional goal vector
            temperature: Sampling temperature

        Returns:
            next_token: Sampled next token ID (batch,)
            logits: Output logits for last position (batch, vocab_size)
            entropy: Entropy of output distribution (scalar)
        """
        # forward pass
        logits, entropies = self.forward(
            input_ids,
            goal_state=goal_state,
            return_entropy=True
        )

        # Get logits for last position
        next_token_logits = logits[:, -1, :] / temperature  # (batch, vocab_size)

        # Sample next token
        probs = f.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (batch,)

        # Entropy of last position
        if entropies is not None:
            entropy = entropies[:, -1].mean().item()
        else:
            # Fallback: compute entropy from the last-position logits directly
            probs_last = f.softmax(next_token_logits, dim=-1)
            log_probs_last = f.log_softmax(next_token_logits, dim=-1)
            entropy = -(probs_last * log_probs_last).sum(dim=-1).mean().item()

        return next_token, next_token_logits, entropy

    def __repr__(self) -> str:
        return (
            f"TransformerDecoder(vocab={self.config.vocab_size}, "
            f"dim={self.embedding_dim}, layers={self.config.num_layers})"
        )
