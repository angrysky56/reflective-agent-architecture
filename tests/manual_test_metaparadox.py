import unittest
from unittest.mock import MagicMock

from src.server import CognitiveWorkspace


class TestMetaParadox(unittest.TestCase):
    def test_resolve_meta_paradox(self):
        # Mock workspace
        workspace = MagicMock(spec=CognitiveWorkspace)

        # Mock deconstruct return
        workspace.deconstruct.return_value = {
            "root_node": {"id": "root_1"},
            "components": [
                {"id": "comp_1", "content": "Thesis: Validator says Yes"},
                {"id": "comp_2", "content": "Antithesis: Critique says No"}
            ]
        }

        # Mock hypothesize return
        workspace.hypothesize.return_value = {
            "hypothesis_id": "hypo_1",
            "hypothesis": "Synthesis: Validator lacks depth metric"
        }

        # Mock synthesize return
        workspace.synthesize.return_value = {
            "synthesis": "Resolution: Update Validator to check for depth.",
            "critique": "Valid resolution."
        }

        # Bind the method to the mock (simulating the real method on the mock object)
        # Actually, we want to test the logic inside resolve_meta_paradox, so we should instantiate a real workspace with mocked dependencies?
        # Or just trust the integration test.
        # Let's mock the internal calls of the real method.

        # Real method logic requires self.deconstruct, self.hypothesize, self.synthesize.
        # We can attach the real method to the mock.
        workspace.resolve_meta_paradox = CognitiveWorkspace.resolve_meta_paradox.__get__(workspace, CognitiveWorkspace)

        # Execute
        result = workspace.resolve_meta_paradox("Validator says Yes but Critique says No")

        # Verify
        print("Result:", result)
        workspace.deconstruct.assert_called_once()
        workspace.hypothesize.assert_called_once()
        workspace.synthesize.assert_called_once()

        self.assertEqual(result["message"], "Meta-Paradox resolved.")
        self.assertIn("Resolution", result["resolution"])

if __name__ == "__main__":
    unittest.main()
