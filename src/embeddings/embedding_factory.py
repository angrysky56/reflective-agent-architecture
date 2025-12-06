import logging
import os
from typing import Optional

import torch

from src.embeddings.base_embedding_provider import BaseEmbeddingProvider
from src.embeddings.lmstudio_embedding_provider import LMStudioEmbeddingProvider
from src.embeddings.ollama_embedding_provider import OllamaEmbeddingProvider
from src.embeddings.sentence_transformer_provider import SentenceTransformerProvider

logger = logging.getLogger(__name__)

# Global cache for embedding providers (singleton pattern)
# Key: (provider_name, model_name, device)
_PROVIDER_CACHE: dict[tuple[str, str, str], BaseEmbeddingProvider] = {}


class EmbeddingFactory:
    """Factory to create embedding providers based on configuration."""

    @staticmethod
    def create(
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ) -> BaseEmbeddingProvider:
        """
        Create an embedding provider (cached).

        Args:
            provider_name: Provider type ('sentence-transformers', 'ollama', 'lm_studio')
            model_name: Model name/path
            device: Device to use ('cpu', 'cuda', 'mps')
            **kwargs: Additional provider-specific arguments

        Returns:
            Embedding provider instance (cached singleton per config)
        """
        provider_name = provider_name or os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")
        model_name = model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")

        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Normalize provider name
        provider_name = provider_name.lower()

        # Check cache first
        cache_key = (provider_name, model_name, device)
        if cache_key in _PROVIDER_CACHE:
            logger.debug(f"Using cached embedding provider: {provider_name}/{model_name}")
            return _PROVIDER_CACHE[cache_key]

        logger.info(f"Creating embedding provider: {provider_name} with model: {model_name} on {device}")

        # Create provider
        provider: BaseEmbeddingProvider
        if provider_name == "sentence-transformers" or provider_name == "sentencetransformers":
            provider = EmbeddingFactory._create_sentence_transformer(model_name, device, **kwargs)

        elif provider_name == "ollama":
            base_url = kwargs.get("base_url") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            provider = OllamaEmbeddingProvider(model_name, base_url=base_url, device=device)

        elif provider_name == "lm_studio" or provider_name == "lmstudio":
            base_url = kwargs.get("base_url") or os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
            api_key = kwargs.get("api_key") or os.getenv("LMSTUDIO_API_KEY", "lm-studio")
            provider = LMStudioEmbeddingProvider(model_name, base_url=base_url, api_key=api_key, device=device)

        else:
            logger.warning(f"Unknown embedding provider '{provider_name}', defaulting to sentence-transformers")
            provider = EmbeddingFactory._create_sentence_transformer(model_name, device, **kwargs)

        # Store in cache
        _PROVIDER_CACHE[cache_key] = provider
        return provider

    @staticmethod
    def _create_sentence_transformer(model_name: str, device: str, **kwargs) -> SentenceTransformerProvider:
        """
        Create SentenceTransformer provider with optimizations.

        Automatically detects and optimizes for:
        - Qwen models: Uses flash_attention_2 if available on GPU
        - GPU acceleration when available
        """
        is_qwen = "qwen" in model_name.lower()
        has_gpu = device == "cuda"

        model_kwargs = {}
        tokenizer_kwargs = {}

        if is_qwen and has_gpu:
            try:
                # Try flash_attention_2 for GPU acceleration
                model_kwargs = {"attn_implementation": "flash_attention_2", "device_map": "auto"}
                tokenizer_kwargs = {"padding_side": "left"}
                logger.info("Attempting to use flash_attention_2 for Qwen model")
            except Exception as e:
                # Fallback to standard attention
                logger.info(f"flash_attention_2 not available, using standard attention: {e}")
                model_kwargs = {}
                tokenizer_kwargs = {"padding_side": "left"}
        elif is_qwen:
            # CPU mode - just use left padding
            tokenizer_kwargs = {"padding_side": "left"}

        # Merge with user-provided kwargs
        model_kwargs.update(kwargs.get("model_kwargs", {}))
        tokenizer_kwargs.update(kwargs.get("tokenizer_kwargs", {}))

        try:
            return SentenceTransformerProvider(
                model_name,
                device=device,
                model_kwargs=model_kwargs if model_kwargs else None,
                tokenizer_kwargs=tokenizer_kwargs if tokenizer_kwargs else None
            )
        except Exception as e:
            # If optimizations fail, try basic initialization
            logger.warning(f"Failed to create optimized SentenceTransformer, using basic: {e}")
            return SentenceTransformerProvider(model_name, device=device)
