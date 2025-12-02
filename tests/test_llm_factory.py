import os
import sys
import unittest
from unittest.mock import MagicMock

# Debug path
print(f"Current CWD: {os.getcwd()}")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(f"Adding to path: {project_root}")
sys.path.insert(0, project_root)

# Mock external dependencies BEFORE importing src modules
# We need to mock the modules that might be installed but are old, or not installed
mock_openai = MagicMock()
mock_openai.AsyncOpenAI = MagicMock()
sys.modules["openai"] = mock_openai

sys.modules["anthropic"] = MagicMock()
sys.modules["google"] = MagicMock()
sys.modules["google.generativeai"] = MagicMock()
sys.modules["google.generativeai.types"] = MagicMock()
sys.modules["huggingface_hub"] = MagicMock()
sys.modules["huggingface_hub.constants"] = MagicMock()
sys.modules["ollama"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["chromadb"] = MagicMock()
sys.modules["neo4j"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.nn.functional"] = MagicMock()
sys.modules["numpy"] = MagicMock()

try:
    from src.llm.anthropic_provider import AnthropicProvider
    from src.llm.factory import LLMFactory
    from src.llm.gemini_provider import GeminiProvider
    from src.llm.huggingface_provider import HuggingFaceProvider
    from src.llm.ollama_provider import OllamaProvider
    from src.llm.openai_provider import OpenAIProvider
    from src.llm.openrouter_provider import OpenRouterProvider
    from src.llm.provider import BaseLLMProvider
except ImportError as e:
    print(f"ImportError during setup: {e}")
    # Print sys.path
    print("sys.path:", sys.path)
    raise

class TestLLMFactory(unittest.TestCase):
    def setUp(self):
        # Reset environment variables
        keys_to_remove = ["LLM_PROVIDER", "LLM_MODEL", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "HF_TOKEN"]
        for key in keys_to_remove:
            if key in os.environ:
                del os.environ[key]

    def test_default_provider(self):
        # Default should be Ollama
        provider = LLMFactory.create_provider()
        self.assertIsInstance(provider, OllamaProvider)

    def test_ollama_provider(self):
        os.environ["LLM_PROVIDER"] = "ollama"
        provider = LLMFactory.create_provider()
        self.assertIsInstance(provider, OllamaProvider)

    def test_openai_provider(self):
        os.environ["LLM_PROVIDER"] = "openai"
        os.environ["OPENAI_API_KEY"] = "test_key"
        provider = LLMFactory.create_provider()
        self.assertIsInstance(provider, OpenAIProvider)

    def test_anthropic_provider(self):
        os.environ["LLM_PROVIDER"] = "anthropic"
        os.environ["ANTHROPIC_API_KEY"] = "test_key"
        provider = LLMFactory.create_provider()
        self.assertIsInstance(provider, AnthropicProvider)

    def test_gemini_provider(self):
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_API_KEY"] = "test_key"
        provider = LLMFactory.create_provider()
        self.assertIsInstance(provider, GeminiProvider)

    def test_huggingface_provider(self):
        os.environ["LLM_PROVIDER"] = "huggingface"
        os.environ["HF_TOKEN"] = "test_token"
        provider = LLMFactory.create_provider()
        self.assertIsInstance(provider, HuggingFaceProvider)

    def test_lm_studio_provider(self):
        os.environ["LLM_PROVIDER"] = "lm_studio"
        # LM Studio uses OpenAIProvider
        provider = LLMFactory.create_provider()
        self.assertIsInstance(provider, OpenAIProvider)

    def test_openrouter_provider(self):
        os.environ["LLM_PROVIDER"] = "openrouter"
        os.environ["OPENROUTER_API_KEY"] = "test_key"
        provider = LLMFactory.create_provider()
        self.assertIsInstance(provider, OpenRouterProvider)

if __name__ == "__main__":
    unittest.main()
