from unittest.mock import MagicMock, patch

import pytest
import torch

from src.director.director_core import DirectorMVP
from src.director.matrix_monitor import MatrixMonitorConfig


def test_teach_cognitive_state():
    """Test the teach_state capability."""
    # Setup Director with mock monitor
    config = MagicMock()
    config.device = "cpu"
    config.matrix_monitor_config = MatrixMonitorConfig()

    director = DirectorMVP(config)

    # 1. Try teaching without a thought (should fail)
    assert director.teach_state("Focused") is False

    # 2. Simulate a thought
    fake_attention = torch.rand((1, 8, 16, 16))
    director.monitor_thought_process(fake_attention)

    # 3. Teach state (should succeed)
    assert director.teach_state("Focused") is True

    # 4. Verify it was registered
    labels = director.get_known_states()
    assert "Focused" in labels.values()

def test_visualize_thought():
    """Test ASCII visualization."""
    config = MagicMock()
    config.device = "cpu"
    director = DirectorMVP(config)

    # 1. No thought
    assert "No recent thought" in director.visualize_last_thought()

    # 2. With thought
    fake_attention = torch.rand((1, 8, 16, 16))
    director.monitor_thought_process(fake_attention)
    vis = director.visualize_last_thought()
    assert "Cognitive Topology" in vis
    assert "(8x8)" in vis

# Note: We skip mocking Neo4j for inspect_knowledge_graph as it requires a live DB or complex mocking.
# We assume the method logic is correct based on code review and will rely on integration tests if needed.
