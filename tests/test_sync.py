
import logging
from unittest.mock import MagicMock

import torch

from src.integration.cwd_raa_bridge import BridgeConfig, CWDRAABridge
from src.manifold.hopfield_network import HopfieldConfig, ModernHopfieldNetwork

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sync():
    logger.info("Starting sync test...")

    # 1. Setup Mock CWD Server and Workspace
    mock_cwd_server = MagicMock()
    mock_workspace = MagicMock()

    # Create a dummy tool library
    tool_library = {
        "tool_1": {
            "name": "Test Tool 1",
            "embedding": [0.1] * 64, # 64-dim embedding
            "pattern": "some pattern"
        },
        "tool_2": {
            "name": "Test Tool 2",
            "embedding": [0.9] * 64,
            "pattern": "another pattern"
        }
    }
    mock_workspace.tool_library = tool_library
    mock_cwd_server.workspace = mock_workspace

    # 2. Setup Manifold
    config = HopfieldConfig(embedding_dim=64, device="cpu")
    manifold = ModernHopfieldNetwork(config)

    # 3. Setup Bridge
    bridge_config = BridgeConfig(embedding_dim=64, device="cpu")
    bridge = CWDRAABridge(
        cwd_server=mock_cwd_server,
        raa_director=MagicMock(),
        manifold=manifold,
        config=bridge_config
    )

    # 4. Run Sync
    logger.info("Running sync_tools_to_manifold...")
    count = bridge.sync_tools_to_manifold()

    # 5. Verify
    logger.info(f"Synced {count} tools.")

    # Check Manifold patterns
    num_patterns = manifold.num_patterns
    logger.info(f"Manifold has {num_patterns} patterns.")

    if count == 2 and num_patterns == 2:
        logger.info("SUCCESS: All tools synced.")
    else:
        logger.error(f"FAILURE: Expected 2 tools, got {count} synced and {num_patterns} in Manifold.")

    # Check metadata
    if len(manifold.pattern_metadata) == 2:
        logger.info(f"Metadata: {manifold.pattern_metadata}")
    else:
        logger.error("FAILURE: Metadata missing.")

if __name__ == "__main__":
    test_sync()
