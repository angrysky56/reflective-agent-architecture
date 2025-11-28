import asyncio
import json
from unittest.mock import MagicMock

import torch

from src.server import RAAServerContext, call_tool, server_context


# Mock Workspace
class MockWorkspace:
    def _llm_generate(self, system, user):
        return json.dumps({
            "state_fragment": "Python CLI Environment",
            "agent_fragment": "Debugger Persona",
            "action_fragment": "Analyze Error"
        })

# Mock Bridge & Mapper
class MockMapper:
    def __init__(self):
        self.embedding_model = MagicMock()
        self.embedding_model.encode.return_value = torch.randn(384) # Match dim

class MockBridge:
    def __init__(self):
        self.embedding_mapper = MockMapper()

# Setup Context
async def test_deconstruct():
    # Initialize context components
    server_context.workspace = MockWorkspace()
    server_context.bridge = MockBridge()
    # server_context.device = "cpu" # Removed: device is a property, set via raa_context below

    # Initialize Manifold & Precuneus (Real ones)
    from src.integration.precuneus import PrecuneusIntegrator
    from src.manifold import HopfieldConfig, Manifold

    config = HopfieldConfig(embedding_dim=384)
    server_context.raa_context = {
        "manifold": Manifold(config),
        "precuneus": PrecuneusIntegrator(384),
        "bridge": server_context.bridge,
        "device": "cpu",
        "agent_factory": MagicMock()
    }

    # Call Tool
    args = {"problem": "Fix the syntax error in server.py"}
    result = await call_tool("deconstruct", args)

    print("Result:", result[0].text)

if __name__ == "__main__":
    asyncio.run(test_deconstruct())
