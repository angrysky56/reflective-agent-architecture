
import sys
import os
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import json
import torch

# Add project root to path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)

# Mock dependencies BEFORE importing server
sys.modules["neo4j"] = MagicMock()
sys.modules["chromadb"] = MagicMock()
sys.modules["chromadb.config"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["ollama"] = MagicMock()
sys.modules["mcp"] = MagicMock()
sys.modules["mcp.server"] = MagicMock()
sys.modules["mcp.types"] = MagicMock()

# Mock TextContent
class MockTextContent:
    def __init__(self, type, text):
        self.type = type
        self.text = text
sys.modules["mcp.types"].TextContent = MockTextContent

# Import server
from src import server

class TestPhase2Tools(unittest.IsolatedAsyncioTestCase):
    async def test_orthogonal_dimensions_analyzer(self):
        # Setup Mock Context
        mock_ctx = MagicMock()
        mock_workspace = MagicMock()
        mock_ctx.workspace = mock_workspace
        
        # Mock ContinuityField
        mock_continuity_field = MagicMock()
        mock_workspace.continuity_field = mock_continuity_field
        
        # Mock Embedding Model
        mock_embedding_model = MagicMock()
        mock_embedding_model.encode.side_effect = lambda x: torch.tensor([0.1, 0.2, 0.3]) if x == "concept_a" else torch.tensor([0.4, 0.5, 0.6])
        mock_workspace.embedding_model = mock_embedding_model
        
        # Mock Config
        mock_config = MagicMock()
        mock_config.llm_model = "test-model"
        mock_workspace.config = mock_config
        
        # Mock Ollama
        server.ollama.chat.return_value = {
            "message": {"content": "Qualitative analysis result"}
        }
        
        # Patch get_raa_context
        with patch("src.server.get_raa_context", return_value=mock_ctx):
            # Call tool
            arguments = {
                "concept_a": "concept_a",
                "concept_b": "concept_b",
                "context": "test context"
            }
            
            # Need to mock OrthogonalDimensionsAnalyzer because it's instantiated inside call_tool
            # But we can also let it run if we mocked its dependencies (continuity_field) correctly.
            # However, analyze_vectors might fail if we don't mock it or if inputs aren't right.
            # Let's mock the class to be safe and focus on the integration logic
            with patch("src.server.OrthogonalDimensionsAnalyzer") as MockAnalyzer:
                mock_analyzer_instance = MockAnalyzer.return_value
                mock_analyzer_instance.analyze_vectors.return_value = {"orthogonality": 0.5}
                mock_analyzer_instance.construct_analysis_prompt.return_value = "test prompt"
                mock_analyzer_instance.SYSTEM_PROMPT = "system prompt"
                
                result = await server.call_tool("orthogonal_dimensions_analyzer", arguments)
                
                # Verify result
                self.assertEqual(len(result), 1)
                content = json.loads(result[0].text)
                
                self.assertEqual(content["concepts"]["a"], "concept_a")
                self.assertEqual(content["vector_analysis"]["orthogonality"], 0.5)
                self.assertEqual(content["qualitative_analysis"], "Qualitative analysis result")
                
                print("Orthogonal Dimensions Analyzer tool verified successfully!")

if __name__ == "__main__":
    unittest.main()
