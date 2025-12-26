import logging
from typing import List, Union

import numpy as np

from reflective_agent_architecture.embeddings.base_embedding_provider import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class LMStudioEmbeddingProvider(BaseEmbeddingProvider):
    """LM Studio embedding provider using OpenAI-compatible API."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
        device: str = "cpu"  # LM Studio manages device internally
    ):
        """
        Initialize LM Studio embedding provider.

        Args:
            model_name: Name of the model loaded in LM Studio
            base_url: Base URL for LM Studio API
            api_key: API key (default works for LM Studio)
            device: Placeholder for compatibility
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.device = device
        self._embedding_dim = None

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"LM Studio embedding provider initialized: {model_name} at {base_url}")
        except ImportError:
            logger.error("openai package not installed. Install with: pip install openai")
            raise

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Encode sentences using LM Studio embeddings API.

        Args:
            sentences: Single sentence or list of sentences
            batch_size: Batch size for processing
            show_progress_bar: Whether to show progress (not implemented)
            convert_to_numpy: Whether to convert to numpy array

        Returns:
            Embeddings as numpy array
        """
        # Handle single string
        is_single = isinstance(sentences, str)
        if is_single:
            sentences = [sentences]

        try:
            # LM Studio uses OpenAI-compatible embeddings endpoint
            response = self.client.embeddings.create(
                model=self.model_name,
                input=sentences
            )

            embeddings = [item.embedding for item in response.data]

            # Cache dimension from first successful call
            if self._embedding_dim is None and embeddings:
                self._embedding_dim = len(embeddings[0])

            if convert_to_numpy:
                embeddings = np.array(embeddings, dtype=np.float32)
                if is_single:
                    return embeddings[0]
                return embeddings
            else:
                if is_single:
                    return embeddings[0]
                return embeddings

        except Exception as e:
            logger.error(f"LM Studio embedding request failed: {e}")
            raise RuntimeError(f"Failed to get embedding from LM Studio: {e}")

    def get_sentence_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        if self._embedding_dim is None:
            # Get dimension by encoding a test sentence
            _ = self.encode("test", convert_to_numpy=False)
        return self._embedding_dim

    def to(self, device: str):
        """LM Studio manages device internally, this is a no-op for compatibility."""
        self.device = device
        return self
