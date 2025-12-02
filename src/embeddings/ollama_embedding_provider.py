import logging
from typing import List, Union

import numpy as np
import requests

from src.embeddings.base_embedding_provider import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Ollama embedding provider for local embedding models."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        device: str = "cpu"  # Ollama manages device internally
    ):
        """
        Initialize Ollama embedding provider.

        Args:
            model_name: Name of the Ollama model (e.g., 'nomic-embed-text', 'mxbai-embed-large')
            base_url: Base URL for Ollama API
            device: Placeholder for compatibility, Ollama manages device
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.device = device
        self._embedding_dim = None

        # Verify Ollama is accessible and model exists
        try:
            self._verify_ollama_connection()
            logger.info(f"Ollama embedding provider initialized: {model_name} at {base_url}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama embedding provider: {e}")
            raise

    def _verify_ollama_connection(self):
        """Verify Ollama server is accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.RequestException as e:
            raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}: {e}")

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Encode sentences using Ollama embeddings API.

        Args:
            sentences: Single sentence or list of sentences
            batch_size: Batch size for processing (Ollama processes one at a time)
            show_progress_bar: Whether to show progress (not implemented for Ollama)
            convert_to_numpy: Whether to convert to numpy array

        Returns:
            Embeddings as numpy array
        """
        # Handle single string
        is_single = isinstance(sentences, str)
        if is_single:
            sentences = [sentences]

        embeddings = []

        for sentence in sentences:
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.model_name,
                        "prompt": sentence
                    },
                    timeout=30
                )
                response.raise_for_status()
                embedding = response.json()["embedding"]
                embeddings.append(embedding)

                # Cache dimension from first successful call
                if self._embedding_dim is None:
                    self._embedding_dim = len(embedding)

            except requests.RequestException as e:
                logger.error(f"Ollama embedding request failed: {e}")
                raise RuntimeError(f"Failed to get embedding from Ollama: {e}")

        if convert_to_numpy:
            embeddings = np.array(embeddings, dtype=np.float32)
            if is_single:
                return embeddings[0]
            return embeddings
        else:
            if is_single:
                return embeddings[0]
            return embeddings

    def get_sentence_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        if self._embedding_dim is None:
            # Get dimension by encoding a test sentence
            _ = self.encode("test", convert_to_numpy=False)
        return self._embedding_dim

    def to(self, device: str):
        """Ollama manages device internally, this is a no-op for compatibility."""
        self.device = device
        return self
