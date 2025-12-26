import logging
from typing import Any, List, Optional, Union

import numpy as np

from reflective_agent_architecture.embeddings.base_embedding_provider import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """SentenceTransformers embedding provider."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        model_kwargs: Optional[dict] = None,
        tokenizer_kwargs: Optional[dict] = None
    ):
        """
        Initialize SentenceTransformer provider.

        Args:
            model_name: Name/path of the sentence-transformers model
            device: Device to use ('cpu', 'cuda', 'mps')
            model_kwargs: Additional kwargs for model initialization
            tokenizer_kwargs: Additional kwargs for tokenizer
        """
        self.model_name = model_name
        self.device = device
        self.model_kwargs = model_kwargs or {}
        self.tokenizer_kwargs = tokenizer_kwargs or {}

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                model_name,
                model_kwargs=self.model_kwargs if self.model_kwargs else None,
                tokenizer_kwargs=self.tokenizer_kwargs if self.tokenizer_kwargs else None
            )
            self.model.to(device)
            logger.info(f"Loaded SentenceTransformer: {model_name} on {device}")

        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        **kwargs
    ) -> Union[np.ndarray, List[np.ndarray], Any]:
        """Encode sentences using SentenceTransformer."""
        # Handle device argument to avoid duplicates with kwargs
        device = kwargs.pop("device", self.device)

        # If converting to tensor, disable numpy conversion to avoid conflicts
        if kwargs.get("convert_to_tensor", False):
            convert_to_numpy = False

        return self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=convert_to_numpy,
            device=device,
            **kwargs
        )

    def get_sentence_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def to(self, device: str):
        """Move model to device."""
        self.device = device
        self.model.to(device)
        return self
