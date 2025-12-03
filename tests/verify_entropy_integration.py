import json
import os
import sys
from unittest.mock import MagicMock, patch

# Add src to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

print(f"CWD: {os.getcwd()}")
print(f"Project Root: {project_root}")
print(f"Sys Path: {sys.path}")

try:
    from src.server import CognitiveWorkspace
    from src.substrate.entropy import CognitiveState
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)


import numpy as np


def test_tool_usage_tracking():
    print("Testing Tool Usage Tracking...")

    # Mock dependencies to avoid full initialization
    with patch('src.server.GraphDatabase'), \
         patch('src.server.chromadb'), \
         patch('src.server.LLMFactory'), \
         patch('src.server.EmbeddingFactory') as MockEmbeddingFactory, \
         patch('src.server.MetabolicLedger'), \
         patch('src.server.ContinuityService'), \
         patch('src.server.SystemGuideNodes'), \
         patch('src.server.CuriosityModule'), \
         patch('src.server.load_dotenv'):

        # Configure Embedding Mock
        mock_embedding_provider = MockEmbeddingFactory.create.return_value
        mock_embedding_provider.get_sentence_embedding_dimension.return_value = 384
        mock_embedding_provider.encode.return_value = np.zeros(384)

        mock_config = MagicMock()
        mock_config.neo4j_uri = "bolt://localhost:7687"
        mock_config.neo4j_user = "neo4j"
        mock_config.neo4j_password = "password"
        mock_config.chroma_path = "./chroma_data"
        mock_config.llm_provider = "openrouter"
        mock_config.llm_model = "google/gemini-2.0-flash-exp:free"
        mock_config.embedding_provider = "sentence-transformers"
        mock_config.embedding_model = "BAAI/bge-large-en-v1.5"

        workspace = CognitiveWorkspace(mock_config)

        # Verify initial state
        assert len(workspace.tool_usage_buffer) == 0
        print("Initial buffer empty: OK")

        # 1. Call a tool multiple times (Low Entropy)
        print("Simulating repetitive tool usage (Low Entropy)...")
        for _ in range(10):
            try:
                workspace._read_file("non_existent_file")
            except:
                pass # Ignore actual file errors

        assert len(workspace.tool_usage_buffer) == 10
        assert all(t == "read_file" for t in workspace.tool_usage_buffer)
        print(f"Buffer: {workspace.tool_usage_buffer}")

        # Check Entropy
        entropy = workspace.entropy_monitor.current_entropy
        state = workspace.entropy_monitor.state
        print(f"Current Entropy: {entropy}, State: {state}")
        assert entropy == 0.0
        assert state == CognitiveState.EXPLORE # Low entropy triggers Explore
        print("Low Entropy Triggered Explore: OK")

        # 2. Call different tools (High Entropy)
        print("Simulating diverse tool usage (High Entropy)...")
        tools = ["list_directory", "search_codebase", "inspect_codebase"]
        for i in range(10):
            tool = tools[i % 3]
            try:
                if tool == "list_directory":
                    workspace._list_directory(".")
                elif tool == "search_codebase":
                    workspace._search_codebase("test")
                elif tool == "inspect_codebase":
                    workspace._inspect_codebase("scan")
            except:
                pass

        print(f"Buffer: {workspace.tool_usage_buffer}")

        # Check Entropy
        entropy = workspace.entropy_monitor.current_entropy
        state = workspace.entropy_monitor.state
        print(f"Current Entropy: {entropy}, State: {state}")
        assert entropy > 1.5 # Should be higher
        # Note: State might stick to EXPLORE due to hysteresis or transition to FOCUS if high enough

        print("Integration Verification Complete!")

        # 3. Verify Historical Baselines
        print("Verifying Historical Baselines...")
        status = json.loads(workspace._get_cognitive_state())
        assert "baseline" in status
        assert "trend" in status
        print(f"Baseline: {status['baseline']}, Trend: {status['trend']}")
        print("Historical Baselines: OK")

        # 4. Verify State-Based Search Behavior
        print("Verifying State-Based Search Behavior...")
        with patch('subprocess.run') as mock_run:
            # Generate dummy output > 100 lines
            long_output = "\n".join([f"match {i}" for i in range(300)])
            mock_run.return_value.stdout = long_output
            mock_run.return_value.returncode = 0

            # Test EXPLORE state (default/low entropy)
            workspace.entropy_monitor.state = CognitiveState.EXPLORE
            result_explore = workspace._search_codebase("test")
            assert "truncated due to explore state" in result_explore
            # Should show 200 lines
            assert "match 199" in result_explore
            assert "match 200" not in result_explore
            print("Explore State Search (200 limit): OK")

            # Test FOCUS state
            workspace.entropy_monitor.state = CognitiveState.FOCUS
            result_focus = workspace._search_codebase("test")
            assert "truncated due to focus state" in result_focus
            # Should show 20 lines
            assert "match 19" in result_focus
            assert "match 20" not in result_focus
            print("Focus State Search (20 limit): OK")

if __name__ == "__main__":
    test_tool_usage_tracking()
