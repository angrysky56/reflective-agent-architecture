import logging
import os
from typing import Optional

from src.llm.anthropic_provider import AnthropicProvider
from src.llm.gemini_provider import GeminiProvider
from src.llm.huggingface_provider import HuggingFaceProvider
from src.llm.ollama_provider import OllamaProvider
from src.llm.openai_provider import OpenAIProvider
from src.llm.openrouter_provider import OpenRouterProvider
from src.llm.provider import BaseLLMProvider

logger = logging.getLogger(__name__)


class LLMFactory:
    """Factory to create LLM providers based on configuration."""

    @staticmethod
    def create_provider(
        provider_name: Optional[str] = None, model_name: Optional[str] = None
    ) -> BaseLLMProvider:
        provider_name = provider_name or os.getenv("LLM_PROVIDER", "openrouter").lower()
        model_name = (
            model_name
            if model_name is not None
            else os.getenv("LLM_MODEL", "google/gemini-3-pro-preview")
        )

        # Ensure model_name is never None (fallback to default if getenv returns None)
        if model_name is None:
            model_name = "google/gemini-3-pro-preview"

        logger.info(f"Initializing LLM provider: {provider_name} with model: {model_name}")

        if provider_name == "ollama":
            return OllamaProvider(model_name)
        elif provider_name == "openai" or provider_name == "lm_studio":
            return OpenAIProvider(model_name)
        elif provider_name == "openrouter":
            return OpenRouterProvider(model_name)
        elif provider_name == "anthropic":
            return AnthropicProvider(model_name)
        elif provider_name == "gemini":
            return GeminiProvider(model_name)
        elif provider_name == "huggingface":
            return HuggingFaceProvider(model_name)
        else:
            logger.warning(f"Unknown provider '{provider_name}', defaulting to OpenRouter")
            return OpenRouterProvider(model_name)
