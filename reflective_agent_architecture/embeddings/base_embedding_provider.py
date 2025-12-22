from abc import ABC, abstractmethod
from typing import Any, List, Union

import numpy as np


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        **kwargs: Any,
    ) -> Union[np.ndarray, List[np.ndarray], Any]:
        """
        Encode sentences into embeddings.

        Args:
            sentences: Single sentence or list of sentences to encode
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar
            convert_to_numpy: Whether to convert to numpy array
            **kwargs: Additional arguments passed to underlying model

        Returns:
            Embeddings as numpy array or list of arrays
        """
        pass

    @abstractmethod
    def get_sentence_embedding_dimension(self) -> int:
        """Get the dimension of the sentence embeddings."""
        pass

    @abstractmethod
    def to(self, device: str) -> None:
        """Move model to device (cpu/cuda)."""
        pass
