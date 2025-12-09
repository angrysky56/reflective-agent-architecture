import logging
from typing import Any, List, Union

import numpy as np

from src.embeddings.base_embedding_provider import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class OpenRouterEmbeddingProvider(BaseEmbeddingProvider):
    """OpenRouter embedding provider using OpenAI-compatible API."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        device: str = "cpu",  # OpenRouter manages device internally
    ):
        """
        Initialize OpenRouter embedding provider.

        Args:
            model_name: Name of the model on OpenRouter
            api_key: OpenRouter API key
            base_url: Base URL for OpenRouter API
            device: Placeholder for compatibility
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.device = device
        self._embedding_dim: Union[int, None] = None

        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"OpenRouter embedding provider initialized: {model_name}")
        except ImportError:
            logger.error("openai package not installed. Install with: pip install openai")
            raise

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        **kwargs: Any,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Encode sentences using OpenRouter embeddings API.

        Args:
            sentences: Single sentence or list of sentences
            batch_size: Batch size for processing (not fully valid for API calls but kept for interface)
            show_progress_bar: Whether to show progress (not implemented)
            convert_to_numpy: Whether to convert to numpy array

        Returns:
            Embeddings as numpy array
        """
        # Handle single string
        is_single = isinstance(sentences, str)
        if is_single:
            sentences = [sentences]  # type: ignore

        try:
            # OpenRouter uses OpenAI-compatible embeddings endpoint
            # We can't really batch easily with the API wrapper without custom logic,
            # but usually the API handles list of strings fine.
            # Large lists might hit token limits, but for MVP we send all.
            response = self.client.embeddings.create(model=self.model_name, input=sentences)

            # Ensure correct ordering - OpenAI API usually guarantees it but good to be safe if checking indices
            embeddings = [item.embedding for item in response.data]

            # Cache dimension from first successful call
            if self._embedding_dim is None and embeddings:
                self._embedding_dim = len(embeddings[0])

            if convert_to_numpy:
                embeddings_np = np.array(embeddings, dtype=np.float32)
                if is_single:
                    return embeddings_np[0]
                return embeddings_np
            else:
                if is_single:
                    return embeddings[0]
                return embeddings

        except Exception as e:
            logger.error(f"OpenRouter embedding request failed: {e}")
            raise RuntimeError(f"Failed to get embedding from OpenRouter: {e}")

    def get_sentence_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        if self._embedding_dim is None:
            # Get dimension by encoding a test sentence
            _ = self.encode("test", convert_to_numpy=False)

            # If still None (e.g. failure), default to generic or raise
            if self._embedding_dim is None:
                raise RuntimeError(
                    "Could not determine embedding dimension from OpenRouter provider."
                )

        return self._embedding_dim

    def to(self, device: str) -> None:
        """OpenRouter manages device internally, this is a no-op for compatibility."""
        self.device = device
