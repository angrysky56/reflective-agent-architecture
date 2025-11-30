
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)

# Mock dependencies
sys.modules["neo4j"] = MagicMock()
sys.modules["chromadb"] = MagicMock()
sys.modules["chromadb.config"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.nn.functional"] = MagicMock()
sys.modules["torch.optim"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["ollama"] = MagicMock()

from src.server import CognitiveWorkspace, CWDConfig


class TestVectorConnection(unittest.TestCase):
    @patch("src.server.SentenceTransformer")
    @patch("src.server.ContinuityField")
    def test_initialization(self, MockContinuityField, MockSentenceTransformer):
        # Setup mocks
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        MockSentenceTransformer.return_value = mock_model

        # Initialize Workspace
        config = CWDConfig(neo4j_password="test")
        workspace = CognitiveWorkspace(config)

        # Verify ContinuityField initialization
        MockContinuityField.assert_called_once_with(embedding_dim=768)
        self.assertIsNotNone(workspace.continuity_field)
        print("ContinuityField initialized successfully with dim=768")

if __name__ == "__main__":
    unittest.main()
