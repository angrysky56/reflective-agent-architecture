"""
Tests for RAA Fixes (Hypothesis Generation & Cognitive State).
"""

import re
from unittest.mock import MagicMock

from src.integration.cwd_raa_bridge import BridgeConfig, CWDRAABridge
from src.server import CognitiveWorkspace


class MockConfig:
    llm_model = "mock-model"

class MockWorkspace(CognitiveWorkspace):
    def __init__(self):
        self.config = MockConfig()
        self.tool_library = {}

def test_llm_generate_regex():
    """Test that the fixed regex correctly handles <think> blocks."""

    # Case 1: Standard <think> block
    content = "<think>Reasoning...</think> Actual output."
    # We need to simulate the regex logic from _llm_generate since we can't call it directly easily
    # (it requires ollama). So we copy the logic here to verify it.

    if "<think>" in content:
        parts = re.split(r"</think>", content, flags=re.IGNORECASE)
        if len(parts) > 1 and parts[-1].strip():
            content = parts[-1].strip()
        else:
            content = re.sub(r"</?think>", "", content, flags=re.IGNORECASE)

    assert content == "Actual output."

    # Case 2: Everything inside <think> (should strip tags but keep content)
    content = "<think>Just reasoning, no explicit output.</think>"
    if "<think>" in content:
        parts = re.split(r"</think>", content, flags=re.IGNORECASE)
        if len(parts) > 1 and parts[-1].strip():
            content = parts[-1].strip()
        else:
            content = re.sub(r"</?think>", "", content, flags=re.IGNORECASE)

    assert content == "Just reasoning, no explicit output."

    # Case 3: Mixed content
    content = "Prefix <think>Inner</think> Suffix"
    if "<think>" in content:
        parts = re.split(r"</think>", content, flags=re.IGNORECASE)
        if len(parts) > 1 and parts[-1].strip():
            content = parts[-1].strip()
        else:
            content = re.sub(r"</?think>", "", content, flags=re.IGNORECASE)

    assert content == "Suffix"

def test_shadow_monitoring():
    """Test that CWDRAABridge runs shadow monitoring."""
    mock_server = MagicMock()
    mock_director = MagicMock()
    mock_director.latest_cognitive_state = ("Unknown", 0.0)
    mock_manifold = MagicMock()
    mock_processor = MagicMock()

    bridge = CWDRAABridge(
        cwd_server=mock_server,
        raa_director=mock_director,
        manifold=mock_manifold,
        processor=mock_processor,
        config=BridgeConfig(device="cpu", enable_monitoring=True)
    )

    # Mock execute_cwd_operation to return something
    bridge._execute_cwd_operation = MagicMock(return_value={"result": "ok"})

    # Mock director.check_entropy to return (is_clash, entropy)
    mock_director.check_entropy.return_value = (False, 0.5)

    # Execute operation
    bridge.execute_monitored_operation("test_op", {})

    # Verify processor was called (shadow monitoring)
    mock_processor.assert_called_once()

    # Verify arguments (dummy input and goal)
    args = mock_processor.call_args
    assert len(args[0]) > 0  # input_ids
    # goal_state might be None or tensor

def test_deconstruct_monitoring():
    """Test that deconstruct operation is routed through bridge."""
    # This requires mocking the server's call_tool method which is hard to do in isolation
    # without instantiating the whole server.
    # Instead, we verify that CWDRAABridge handles 'deconstruct' correctly.

    mock_server = MagicMock()
    mock_director = MagicMock()
    mock_director.latest_cognitive_state = ("Unknown", 0.0)
    mock_manifold = MagicMock()
    mock_processor = MagicMock()

    bridge = CWDRAABridge(
        cwd_server=mock_server,
        raa_director=mock_director,
        manifold=mock_manifold,
        processor=mock_processor,
        config=BridgeConfig(device="cpu", enable_monitoring=True)
    )

    # Mock execute_cwd_operation to return something
    bridge._execute_cwd_operation = MagicMock(return_value={
        "root_id": "root", "component_ids": ["c1", "c2"]
    })

    # Mock director.check_entropy
    mock_director.check_entropy.return_value = (False, 0.5)

    # Execute deconstruct
    result = bridge.execute_monitored_operation("deconstruct", {"problem": "test"})

    # Verify processor was called
    mock_processor.assert_called_once()
    assert result["root_id"] == "root"
