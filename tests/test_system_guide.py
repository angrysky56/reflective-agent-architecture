import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.cognition.system_guide import SystemGuideNodes
from src.server import CognitiveWorkspace, CWDConfig


class TestSystemGuideNodes(unittest.TestCase):
    def setUp(self):
        self.mock_driver = MagicMock()
        self.mock_session = MagicMock()
        self.mock_driver.session.return_value.__enter__.return_value = self.mock_session

        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)

        self.system_guide = SystemGuideNodes(self.mock_driver, str(self.root_path))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_create_concept(self):
        result = self.system_guide.create_concept("TestConcept", "Description")
        self.assertIn("created/updated", result)
        self.mock_session.run.assert_called()

    def test_create_bookmark(self):
        result = self.system_guide.create_bookmark("test.py", 10, "print('hello')", "notes")
        self.assertIn("test.py:10", result)
        self.mock_session.run.assert_called()

    def test_scan_codebase(self):
        # Create a dummy python file
        py_file = self.root_path / "test_scan.py"
        py_file.write_text("class TestClass:\n    '''Docstring'''\n    pass\n", encoding="utf-8")

        result = self.system_guide.scan_codebase(".")
        self.assertIn("Created/Updated 1 bookmarks", result)

        # Verify calls
        # Should create concept CodebaseIndex
        # Should create bookmark
        # Should link
        self.assertTrue(self.mock_session.run.call_count >= 3)

class TestInspectCodebaseTool(unittest.TestCase):
    def setUp(self):
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

        self.mock_embedding.return_value.get_sentence_embedding_dimension.return_value = 384

        self.workspace = CognitiveWorkspace(self.config)

        # Mock system_guide to avoid real Neo4j calls in this test
        self.workspace.system_guide = MagicMock()

    def tearDown(self):
        self.patcher_driver.stop()
        self.patcher_chroma.stop()
        self.patcher_llm.stop()
        self.patcher_embedding.stop()
        self.patcher_continuity.stop()
        self.patcher_service.stop()

    def test_inspect_codebase_actions(self):
        # Test scan
        self.workspace._inspect_codebase("scan", path=".")
        self.workspace.system_guide.scan_codebase.assert_called_with(".")

        # Test create_concept
        self.workspace._inspect_codebase("create_concept", name="Test", description="Desc")
        self.workspace.system_guide.create_concept.assert_called_with("Test", "Desc")

        # Test bookmark
        self.workspace._inspect_codebase("bookmark", file="f.py", line=1, snippet="s")
        self.workspace.system_guide.create_bookmark.assert_called_with("f.py", 1, "s", "")

        # Test link
        self.workspace._inspect_codebase("link", concept_name="C", bookmark_id="b:1")
        self.workspace.system_guide.link_bookmark_to_concept.assert_called_with("C", "b:1")

        # Test get_concept
        self.workspace.system_guide.get_concept_details.return_value = {"name": "C"}
        result = self.workspace._inspect_codebase("get_concept", name="C")
        self.assertIn('"name": "C"', result)

if __name__ == '__main__':
    unittest.main()
