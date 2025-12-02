import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Only mock sentence_transformers which is safe to mock early
sys.modules["sentence_transformers"] = MagicMock()


class TestEmbeddingFactory(unittest.TestCase):
    """Test embedding factory creation."""

    def setUp(self):
        """Reset environment before each test."""
        keys_to_remove = ["EMBEDDING_PROVIDER", "EMBEDDING_MODEL",  "OLLAMA_BASE_URL", "LMSTUDIO_BASE_URL"]
        for key in keys_to_remove:
            if key in os.environ:
                del os.environ[key]

    @patch("src.embeddings.embedding_factory.torch.cuda.is_available", return_value=False)
    def test_default_provider(self, mock_cuda):
        """Test default provider is sentence-transformers."""
        from src.embeddings.embedding_factory import EmbeddingFactory
        from src.embeddings.sentence_transformer_provider import SentenceTransformerProvider

        provider = EmbeddingFactory.create()
        self.assertIsInstance(provider, SentenceTransformerProvider)

    @patch("src.embeddings.embedding_factory.torch.cuda.is_available", return_value=False)
    def test_sentence_transformer_provider(self, mock_cuda):
        """Test explicit sentence-transformers provider."""
        from src.embeddings.embedding_factory import EmbeddingFactory
        from src.embeddings.sentence_transformer_provider import SentenceTransformerProvider

        os.environ["EMBEDDING_PROVIDER"] = "sentence-transformers"
        os.environ["EMBEDDING_MODEL"] = "BAAI/bge-small-en-v1.5"

        provider = EmbeddingFactory.create()
        self.assertIsInstance(provider, SentenceTransformerProvider)

    def test_ollama_provider(self):
        """Test Ollama provider creation."""
        from src.embeddings.embedding_factory import EmbeddingFactory
        from src.embeddings.ollama_embedding_provider import OllamaEmbeddingProvider

        os.environ["EMBEDDING_PROVIDER"] = "ollama"
        os.environ["EMBEDDING_MODEL"] = "nomic-embed-text"

        # Mock Ollama connection check
        with patch("src.embeddings.ollama_embedding_provider.requests.get") as mock_get:
            mock_get.return_value.raise_for_status = MagicMock()
            provider = EmbeddingFactory.create()
            self.assertIsInstance(provider, OllamaEmbeddingProvider)

    def test_lmstudio_provider(self):
        """Test LM Studio provider creation."""
        from src.embeddings.embedding_factory import EmbeddingFactory
        from src.embeddings.lmstudio_embedding_provider import LMStudioEmbeddingProvider

        os.environ["EMBEDDING_PROVIDER"] = "lm_studio"
        os.environ["EMBEDDING_MODEL"] = "text-embedding-model"

        provider = EmbeddingFactory.create()
        self.assertIsInstance(provider, LMStudioEmbeddingProvider)


if __name__ == "__main__":
    unittest.main()
