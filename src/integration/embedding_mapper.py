"""
Embedding Mapper: Convert between CWD and RAA embedding spaces

This module handles the alignment between CWD's graph-based representations
and RAA's Hopfield network embeddings.

Strategy:
1. Shared Embedding Model: Both use same base model (sentence-transformers)
2. Projection Layer: Optional learned linear mapping (future enhancement)
3. Node Serialization: Convert graph nodes to text for embedding

Key Functions:
- cwd_node_to_vector: Graph node → embedding vector
- vector_to_cwd_query: Embedding vector → CWD search query
- tool_to_vector: CWD tool → embedding vector
"""

import logging
from typing import Any

import torch
import torch.nn.functional as f

logger = logging.getLogger(__name__)


class EmbeddingMapper:
    """
    Maps between CWD's graph space and RAA's Hopfield embedding space.

    Phase 1 Implementation: Simple text-based embedding using shared model.
    Future: Add learned projection layer for optimal alignment.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        preloaded_model: Any | None = None,
    ):
        """
        Initialize embedding mapper.

        Args:
            embedding_dim: Target embedding dimension for RAA
            model_name: Sentence transformer model for text embeddings
            device: Device for computation ('cpu' or 'cuda')
        """
        self.embedding_dim = embedding_dim
        self.model_name = model_name
        self.device = device

        # Initialize embedding model (lazy loading unless provided)
        self._embedding_model = preloaded_model

        if preloaded_model is not None:
            logger.info(f"EmbeddingMapper using preloaded model (dim={embedding_dim})")
        else:
            logger.info(f"EmbeddingMapper initialized with dim={embedding_dim}, model={model_name}")

    @property
    def embedding_model(self):
        """Lazy load embedding model on first use."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedding_model = SentenceTransformer(self.model_name)
                self._embedding_model.to(self.device)
                logger.info(f"Loaded embedding model: {self.model_name}")
            except ImportError:
                logger.error(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
                raise
        else:
            # Ensure device placement is correct
            try:
                self._embedding_model.to(self.device)  # type: ignore[attr-defined]
            except Exception:
                pass
        return self._embedding_model

    def cwd_node_to_vector(
        self,
        node_id: str,
        node_content: str,
        node_metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """
        Convert CWD thought-node to embedding vector.

        Args:
            node_id: CWD node identifier
            node_content: The thought/content of the node
            node_metadata: Optional metadata (utility score, etc.)

        Returns:
            Normalized embedding vector of shape (embedding_dim,)
        """
        # Serialize node to text
        text = self._serialize_node(node_id, node_content, node_metadata)

        # Embed using model
        with torch.no_grad():
            embedding = self.embedding_model.encode(
                text,
                convert_to_tensor=True,
                device=self.device,
            )

            # Ensure we have a tensor (some providers like Ollama return numpy/list)
            if not isinstance(embedding, torch.Tensor):
                embedding = torch.tensor(embedding, device=self.device)

        # Ensure correct dimension
        if embedding.shape[0] != self.embedding_dim:
            # Resize if needed (should match if model chosen correctly)
            logger.warning(
                f"Embedding dimension mismatch: got {embedding.shape[0]}, "
                f"expected {self.embedding_dim}. Using zero-padding/truncation."
            )
            embedding = self._resize_embedding(embedding)

        # Normalize (critical for Hopfield stability)
        embedding = f.normalize(embedding, p=2, dim=0)

        return embedding

    def tool_to_vector(
        self,
        tool_id: str,
        tool_description: str,
        tool_use_cases: list[str] | None = None,
    ) -> torch.Tensor:
        """
        Convert CWD compressed tool to embedding vector.

        Args:
            tool_id: Tool identifier
            tool_description: What the tool does
            tool_use_cases: When to use this tool

        Returns:
            Normalized embedding vector
        """
        # Serialize tool to text
        text = f"Tool: {tool_description}"
        if tool_use_cases:
            text += f" | Use cases: {', '.join(tool_use_cases)}"

        # Embed
        with torch.no_grad():
            embedding = self.embedding_model.encode(
                text,
                convert_to_tensor=True,
                device=self.device,
            )

        # Ensure dimension and normalize
        if embedding.shape[0] != self.embedding_dim:
            embedding = self._resize_embedding(embedding)
        embedding = f.normalize(embedding, p=2, dim=0)

        return embedding

    def vector_to_cwd_query(
        self,
        vector: torch.Tensor,
        top_k: int = 5,
    ) -> str:
        """
        Convert embedding vector to CWD search query.

        This is the reverse mapping: given a Hopfield pattern,
        find semantically similar CWD nodes.

        Args:
            vector: Embedding vector from Manifold
            top_k: Number of similar terms to include in query

        Returns:
            Search query string for CWD

        Note:
            This is a best-effort approximation. True reverse mapping
            would require a learned decoder (future enhancement).
        """
        # For now, use simple nearest-neighbor in semantic space
        # This requires a vocabulary of concepts (to be built)
        logger.warning(
            "vector_to_cwd_query is a placeholder. "
            "Full implementation requires concept vocabulary."
        )

        # Placeholder: return generic query
        return "find related concepts"

    def _serialize_node(
        self,
        node_id: str,
        content: str,
        metadata: dict[str, Any] | None,
    ) -> str:
        """
        Serialize CWD node to text for embedding.

        Format: "Thought: <content> [Context: <metadata>]"
        """
        text = f"Thought: {content}"

        if metadata:
            # Add relevant metadata
            context_parts = []
            if "goal" in metadata:
                context_parts.append(f"goal={metadata['goal']}")
            if "hypothesis" in metadata:
                context_parts.append(f"hypothesis={metadata['hypothesis']}")

            if context_parts:
                text += f" [Context: {', '.join(context_parts)}]"

        return text

    def _resize_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Resize embedding to target dimension.

        Uses zero-padding if too small, truncation if too large.
        """
        current_dim = embedding.shape[0]

        if current_dim < self.embedding_dim:
            # Zero-pad
            padding = torch.zeros(
                self.embedding_dim - current_dim,
                device=embedding.device,
            )
            return torch.cat([embedding, padding])
        elif current_dim > self.embedding_dim:
            # Truncate
            return embedding[: self.embedding_dim]
        else:
            return embedding

    def compute_similarity(
        self,
        vec1: torch.Tensor,
        vec2: torch.Tensor,
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            vec1, vec2: Embedding vectors

        Returns:
            Similarity score in [0, 1]
        """
        similarity = f.cosine_similarity(
            vec1.unsqueeze(0),
            vec2.unsqueeze(0),
        )
        return similarity.item()
