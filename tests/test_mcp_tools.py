import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.server import CognitiveWorkspace, CWDConfig


class TestMCPTools(unittest.TestCase):
    def setUp(self):
        # Mock config
        self.config = MagicMock(spec=CWDConfig)
        self.config.neo4j_uri = "bolt://localhost:7687"
        self.config.neo4j_user = "neo4j"
        self.config.neo4j_password = "password"
        self.config.chroma_path = "./chroma_data"
        self.config.llm_provider = "openrouter"
        self.config.llm_model = "test-model"
        self.config.embedding_provider = "sentence-transformers"
        self.config.embedding_model = "all-MiniLM-L6-v2"

        # Mock dependencies
        self.patcher_driver = patch('src.server.GraphDatabase.driver')
        self.patcher_chroma = patch('src.server.chromadb.Client')
        self.patcher_llm = patch('src.server.LLMFactory.create_provider')
        self.patcher_embedding = patch('src.server.EmbeddingFactory.create')
        self.patcher_continuity = patch('src.server.ContinuityField')
        self.patcher_service = patch('src.server.ContinuityService')

        self.mock_driver = self.patcher_driver.start()
        self.mock_chroma = self.patcher_chroma.start()
        self.mock_llm = self.patcher_llm.start()
        self.mock_embedding = self.patcher_embedding.start()
        self.patcher_continuity.start()
        self.patcher_service.start()

        # Setup embedding mock
        self.mock_embedding.return_value.get_sentence_embedding_dimension.return_value = 384

        # Initialize workspace
        self.workspace = CognitiveWorkspace(self.config)

    def tearDown(self):
        self.patcher_driver.stop()
        self.patcher_chroma.stop()
        self.patcher_llm.stop()
        self.patcher_embedding.stop()
        self.patcher_continuity.stop()
        self.patcher_service.stop()

    def test_read_file(self):
        """Test reading a file."""
        # Create a temporary file
        test_file = Path("test_read.txt")
        test_file.write_text("Hello World", encoding="utf-8")

        try:
            content = self.workspace._read_file(str(test_file.absolute()))
            self.assertEqual(content, "Hello World")

            # Test non-existent file
            content = self.workspace._read_file("non_existent.txt")
            self.assertTrue(content.startswith("Error: File not found"))
        finally:
            if test_file.exists():
                test_file.unlink()

    def test_list_directory(self):
        """Test listing directory."""
        # Use current directory
        content = self.workspace._list_directory(".")
        self.assertIn("[DIR] tests", content)

        # Let's list the tests directory
        content = self.workspace._list_directory("tests")
        self.assertIn("test_mcp_tools.py", content)

    def test_search_codebase(self):
        """Test searching codebase."""
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy file to search in the temp dir
            test_file = Path(temp_dir) / "test_search.txt"
            test_file.write_text("UniquePattern123", encoding="utf-8")

            # Search in the temp dir
            result = self.workspace._search_codebase("UniquePattern123", temp_dir)
            self.assertIn("test_search.txt", result)
            self.assertIn("UniquePattern123", result)

            # Test no match
            result = self.workspace._search_codebase("NonExistentPattern456", temp_dir)
            self.assertIn("No matches found", result)

if __name__ == '__main__':
    unittest.main()
