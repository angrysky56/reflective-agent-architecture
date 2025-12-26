"""
GloVe Embedding Loader

Loads pretrained GloVe embeddings and provides a torch.nn.Embedding interface.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


class GloVeEmbedding:
    """
    Wrapper for GloVe pretrained embeddings.

    Provides:
    - Loading GloVe vectors from text file
    - torch.nn.Embedding-compatible interface
    - Handling of unknown words (random initialization or zero)
    """

    def __init__(
        self,
        glove_path: str,
        embedding_dim: int = 100,
        unk_strategy: str = "random",  # 'random' or 'zeros'
        device: str = "cpu",
    ):
        """
        Initialize GloVe embeddings.

        Args:
            glove_path: Path to GloVe text file (e.g., glove.6B.100d.txt)
            embedding_dim: Dimension of embeddings (should match file)
            unk_strategy: How to handle unknown words ('random' or 'zeros')
            device: Device to store embeddings on
        """
        self.embedding_dim = embedding_dim
        self.unk_strategy = unk_strategy
        self.device = device

        # Load GloVe vectors
        self.word2idx, self.vectors = self._load_glove(glove_path)
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        # Create nn.Embedding with pretrained weights
        self.embedding = nn.Embedding(
            num_embeddings=len(self.word2idx),
            embedding_dim=embedding_dim,
            padding_idx=None,
        )
        self.embedding.weight.data.copy_(self.vectors)
        self.embedding.weight.requires_grad = False  # Freeze pretrained weights
        self.embedding = self.embedding.to(device)

    def _load_glove(self, glove_path: str) -> tuple[Dict[str, int], torch.Tensor]:
        """
        Load GloVe vectors from text file.

        Returns:
            word2idx: Mapping from word to index
            vectors: Tensor of shape (vocab_size, embedding_dim)
        """
        word2idx = {}
        vectors_list = []

        with open(glove_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                parts = line.strip().split()
                word = parts[0]
                vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)

                word2idx[word] = idx
                vectors_list.append(vector)

        vectors = torch.from_numpy(np.array(vectors_list, dtype=np.float32))
        return word2idx, vectors

    def get_vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.word2idx)

    def get_word_idx(self, word: str) -> Optional[int]:
        """Get index for word, or None if unknown."""
        return self.word2idx.get(word.lower(), None)

    def has_word(self, word: str) -> bool:
        """Check if word exists in vocabulary."""
        return word.lower() in self.word2idx

    def __call__(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for indices (nn.Embedding interface).

        Args:
            indices: Tensor of word indices

        Returns:
            Embeddings tensor
        """
        return self.embedding(indices)

    @property
    def weight(self) -> torch.Tensor:
        """Access embedding weight matrix (for compatibility)."""
        return self.embedding.weight

    def get_word_embedding(self, word: str) -> Optional[torch.Tensor]:
        """
        Get embedding for a single word.

        Args:
            word: Word to look up

        Returns:
            Embedding tensor or None if word not found
        """
        idx = self.get_word_idx(word)
        if idx is None:
            return None
        return self.embedding.weight[idx]

    def __repr__(self) -> str:
        return (
            f"GloVeEmbedding(vocab_size={len(self.word2idx)}, "
            f"dim={self.embedding_dim}, device={self.device})"
        )


def load_glove_embeddings(
    embedding_dim: int = 100,
    data_dir: str = "data/embeddings",
    device: str = "cpu",
) -> GloVeEmbedding:
    """
    Convenience function to load GloVe embeddings.

    Args:
        embedding_dim: Dimension (50, 100, 200, or 300)
        data_dir: Directory containing GloVe files
        device: Device to load embeddings on

    Returns:
        GloVeEmbedding instance
    """
    glove_path = Path(data_dir) / f"glove.6B.{embedding_dim}d.txt"

    if not glove_path.exists():
        raise FileNotFoundError(
            f"GloVe file not found: {glove_path}\n"
            f"Download from: http://nlp.stanford.edu/data/glove.6B.zip"
        )

    return GloVeEmbedding(
        glove_path=str(glove_path),
        embedding_dim=embedding_dim,
        device=device,
    )
