from unittest.mock import MagicMock

import pytest
import torch

from src.director.director_core import DirectorConfig, DirectorMVP
from src.director.matrix_monitor import MatrixMonitorConfig
from src.integration.cwd_raa_bridge import BridgeConfig, CWDRAABridge
from src.manifold import HopfieldConfig, ModernHopfieldNetwork


def test_cognitive_simulation():
    """Verify that operations trigger correct cognitive states."""
    # Setup Director
    monitor_config = MatrixMonitorConfig(device="cpu")
    director_config = DirectorConfig(matrix_monitor_config=monitor_config, device="cpu")

    # Mock Manifold
    manifold = MagicMock()
    director = DirectorMVP(manifold, director_config)

    # Setup Bridge
    bridge_config = BridgeConfig(device="cpu")
    bridge = CWDRAABridge(bridge_config, raa_director=director, manifold=manifold)
    bridge.raa_director = director
    bridge.processor = MagicMock() # Just needs to be not None
    bridge.cwd = MagicMock()
    bridge.history = MagicMock()

    # 1. Test Deconstruct -> Focused
    bridge.execute_monitored_operation("deconstruct", {"problem": "test"})
    state, energy = director.latest_cognitive_state
    print(f"Deconstruct State: {state}")
    assert state == "Focused"

    # 2. Test Hypothesize -> Broad
    bridge.execute_monitored_operation("hypothesize", {"node_a": "1", "node_b": "2"})
    state, energy = director.latest_cognitive_state
    print(f"Hypothesize State: {state}")
    assert state == "Broad"

    # 3. Test Diagnose -> Looping
    bridge.execute_monitored_operation("diagnose_pointer", {})
    state, energy = director.latest_cognitive_state
    print(f"Diagnose State: {state}")
    assert state == "Looping"

if __name__ == "__main__":
    test_cognitive_simulation()
