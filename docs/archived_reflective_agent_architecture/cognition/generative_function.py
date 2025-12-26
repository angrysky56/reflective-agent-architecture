import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class GenerativeFunction:
    """
    The 'Ground of Being' in the TKUI architecture.

    This component acts as the active agent that proposes interventions (thoughts/actions).
    In this implementation, it serves as an adapter that converts natural language
    outputs from the LLM into 'Intervention Vectors' (embeddings) that can be
    processed by the Stereoscopic Engine.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embedding_dim: int = 384):
        """
        Initialize the Generative Function.

        Args:
            model_name: Name of the sentence-transformer model to use.
            embedding_dim: Expected dimension of the embeddings (for validation).
        """
        self.embedding_dim = embedding_dim
        self.model_name = model_name

        try:
            logger.info(f"Initializing GenerativeFunction with model: {model_name}")
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {e}")
            self.model = None

    def text_to_intervention(self, text: str) -> Optional[np.ndarray]:
        """
        Convert natural language text into an intervention vector.

        Args:
            text: The proposed thought or action description.

        Returns:
            np.ndarray: The embedding vector, or None if model failed.
        """
        if not self.model:
            logger.warning("Embedding model not initialized. Cannot generate intervention vector.")
            return None

        try:
            # Generate embedding
            embedding = self.model.encode(text)

            # Ensure it matches expected dimension (if specified)
            if self.embedding_dim and embedding.shape[0] != self.embedding_dim:
                # Simple projection or padding could happen here, but for now just warn
                logger.warning(f"Embedding dimension mismatch. Got {embedding.shape[0]}, expected {self.embedding_dim}")

            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding for text: {e}")
            return None
