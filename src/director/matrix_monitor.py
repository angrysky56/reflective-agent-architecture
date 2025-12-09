"""
Matrix Monitor: Cognitive Proprioception

Extracts topological features from Transformer attention matrices to detect
internal cognitive states (e.g., "Focused", "Looping", "Scattered").

Mechanism:
1. Capture raw attention weights from Processor
2. Downsample to fixed-size "topological thumbnails" (preserving shape)
3. Project to embedding space
4. Query "Self-Manifold" to identify cognitive state
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as f

from ..manifold import HopfieldConfig, ModernHopfieldNetwork

logger = logging.getLogger(__name__)


@dataclass
class MatrixMonitorConfig:
    """Configuration for Matrix Monitor."""

    num_heads: int = 8
    embedding_dim: int = 512
    thumbnail_size: int = 8  # Downsample attention to 8x8 grid
    beta: float = 50.0  # High beta for sharp state classification
    device: str = "cpu"


class MatrixMonitor(nn.Module):
    """
    Monitors the 'topology' of thought by analyzing attention matrices.
    """

    def __init__(self, config: MatrixMonitorConfig):
        super().__init__()
        self.config = config
        self.device = config.device

        # 1. The "Self-State" Manifold
        # Stores patterns representing known cognitive states
        hopfield_config = HopfieldConfig(
            embedding_dim=config.embedding_dim, beta=config.beta, device=config.device
        )
        self.self_manifold = ModernHopfieldNetwork(hopfield_config)

        # Labels for the patterns stored in manifold (index -> label)
        self.state_labels: Dict[int, str] = {}

        # 2. Cognitive Projection Network
        # Compresses the visual topology of attention into an embedding
        # Input: num_heads * thumbnail_size * thumbnail_size
        input_dim = config.num_heads * config.thumbnail_size * config.thumbnail_size

        self.projector = nn.Sequential(
            nn.Linear(input_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.ReLU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        ).to(self.device)

        # Fixed-size pooling to handle variable sequence lengths
        self.pool = nn.AdaptiveAvgPool2d((config.thumbnail_size, config.thumbnail_size))

    def process_attention(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Convert raw variable-size attention matrices into a fixed cognitive embedding.

        Args:
            attention_weights: (Batch, Heads, Seq_Len, Seq_Len)

        Returns:
            Cognitive State Vector: (Batch, Embedding_Dim)
        """
        # 1. Downsample to fixed topology (The "Thumbnail")
        # Keeps the global shape (diagonal, vertical lines) but removes scale
        # Shape: (Batch, Heads, 8, 8)
        thumbnails = self.pool(attention_weights)

        # 2. Flatten
        batch_size = thumbnails.shape[0]
        flat_view = thumbnails.view(batch_size, -1)

        # 3. Project to Manifold space
        cognitive_embedding = self.projector(flat_view)

        # Normalize for Hopfield
        return f.normalize(cognitive_embedding, p=2, dim=-1)

    def register_state(self, attention_weights: torch.Tensor, label: str) -> None:
        """
        Teach the monitor a new cognitive state.

        Usage: When you confirm the AI is 'Confused' or 'Focused', call this.
        """
        with torch.no_grad():
            state_vector = self.process_attention(attention_weights)

            # Store in Self-Manifold
            self.self_manifold.store_pattern(state_vector)

            # Record label
            new_index = self.self_manifold.num_patterns - 1
            self.state_labels[new_index] = label

            logger.info(f"Registered new self-state: '{label}' (Index {new_index})")

    def seed_defaults(self) -> None:
        """
        Seed the monitor with archetypal cognitive states.
        """
        logger.info("Seeding default cognitive states...")

        heads = self.config.num_heads

        # 1. Focused: Sharp diagonal attention (local processing)
        # Shape: (Batch=1, Heads=heads, Seq=16, Seq=16)
        focused = torch.eye(16, device=self.device).unsqueeze(0).unsqueeze(0).repeat(1, heads, 1, 1)
        # Add some noise
        focused = focused + 0.1 * torch.rand_like(focused)
        focused = focused / focused.sum(dim=-1, keepdim=True)
        self.register_state(focused, "Focused")

        # 2. Looping: Strong off-diagonal attention (attending to immediate past repeatedly)
        looping = torch.zeros((1, heads, 16, 16), device=self.device)
        for i in range(16):
            # Attend to i-1, i-2 (looping back)
            if i > 0:
                looping[0, :, i, i - 1] = 1.0
            if i > 1:
                looping[0, :, i, i - 2] = 0.5
        looping = looping + 0.1 * torch.rand_like(looping)
        looping = looping / (looping.sum(dim=-1, keepdim=True) + 1e-6)
        self.register_state(looping, "Looping")

        # 3. Broad: Uniform attention (scanning/diffuse)
        broad = torch.ones((1, heads, 16, 16), device=self.device)
        broad = broad / broad.sum(dim=-1, keepdim=True)
        self.register_state(broad, "Broad")

        logger.info("Seeding complete.")

    def check_state(self, attention_weights: torch.Tensor) -> Tuple[str, float, Dict[str, Any]]:
        """
        Diagnose current cognitive state.

        Returns:
            label: Name of the closest known state (e.g. "Looping")
            energy: Stability of this state (Lower = more confident match)
            diagnostics: Detailed metrics (attention stats, similarities)
        """
        with torch.no_grad():
            # 1. Get current "shape" of thought
            query_state = self.process_attention(attention_weights)

            # 2. Query the Self-Manifold
            patterns = self.self_manifold.get_patterns()
            if patterns.shape[0] == 0:
                return "Unknown (Empty Memory)", 0.0, {}

            # Compute similarity to all known states
            # (Batch, Num_Patterns)
            similarities = torch.matmul(query_state, patterns.T)

            # DEBUG: Log statistics to diagnose matching issues
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Attention Stats: Mean={attention_weights.mean():.4f}, Std={attention_weights.std():.4f}"
                )
                logger.debug(f"Query Norm: {torch.norm(query_state):.4f}")
                logger.debug(f"Similarities: {similarities.tolist()}")

            # Find closest match
            best_score, best_idx = torch.max(similarities, dim=-1)
            best_idx = best_idx.item()

            # Calculate energy (stability)
            energy = self.self_manifold.energy(query_state).item()

            # Retrieve label
            label = self.state_labels.get(best_idx, "Unknown")

            # Compile diagnostics
            diagnostics = {
                "attention_mean": float(attention_weights.mean()),
                "attention_std": float(attention_weights.std()),
                "query_norm": float(torch.norm(query_state)),
                "similarities": similarities.tolist(),
                "best_match_score": float(best_score),
            }

            return label, energy, diagnostics

    def visualize_topology(self, attention_weights: torch.Tensor) -> str:
        """
        (Debug) Generates an ASCII art representation of the attention topology.
        """
        # Average over heads for visualization
        # Shape: (Batch, Seq, Seq) -> (Batch, 1, 8, 8)
        avg_attn = attention_weights.mean(dim=1, keepdim=True)
        thumbnail = self.pool(avg_attn).squeeze()  # 8x8 tensor

        # Simple ASCII map
        chars = " .:-=+*#%@"
        result = "\nCognitive Topology (8x8):\n"
        for row in thumbnail:
            line = ""
            for val in row:
                idx = int(val.item() * 9 * 10)  # Scale up contrast
                idx = min(max(idx, 0), len(chars) - 1)
                line += chars[idx] + " "
            result += line + "\n"
        return result
