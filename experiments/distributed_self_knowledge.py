
import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List

# Ensure root is in path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

from src.compass.adapters import Message
from src.compass.config import IntegratedIntelligenceConfig
from src.compass.integrated_intelligence import IntegratedIntelligence
from src.compass.utils import COMPASSLogger


# Mock Backend
class ComputeBackend:
    def __init__(self):
        self.memory = 0
        self.mode = "CPU"

    def allocate(self, use_cuda: bool):
        if use_cuda:
            raise RuntimeError("CUDA Error: Out of Memory (Simulated). Device mapping failed.")
        self.mode = "CPU"
        self.memory += 1024
        return "Allocated 1024MB on CPU"

class MockTool:
    def __init__(self, name, description, inputSchema):
        self.name = name
        self.description = description
        self.inputSchema = inputSchema

# Mock MCP Client
class MockMCPClient:
    def __init__(self):
        self.backend = ComputeBackend()
        self.tools_map = {
            "run_heavy_computation": {
                "name": "run_heavy_computation",
                "description": "Run a heavy numerical computation. Specify device based on performance needs.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "use_cuda": {"type": "boolean", "description": "Whether to use CUDA GPU acceleration (Attempt for speed)."},
                        "task_desc": {"type": "string", "description": "Description of the calculation"}
                    },
                    "required": ["use_cuda"]
                }
            }
        }
        self.is_initialized = True
        # Mock session for call_tool
        self.session = self
        self.external_mcp = None # Ensure attribute exists

    def get_available_tools(self):
        tools = []
        for name, data in self.tools_map.items():
            tools.append(MockTool(data["name"], data["description"], data["inputSchema"]))
        return tools

    async def call_tool(self, name: str, arguments: dict) -> dict:
        if name == "run_heavy_computation":
            use_cuda = arguments.get("use_cuda", False)
            try:
                result = self.backend.allocate(use_cuda)
                return {"content": [{"type": "text", "text": f"Success: {result}"}]}
            except RuntimeError as e:
                return {"content": [{"type": "text", "text": f"Error: {str(e)}"}]}
        return {"content": [{"type": "text", "text": "Unknown tool"}]}

    async def initialize(self):
        pass

# Initialize Components
async def run_experiment():
    logging.basicConfig(level=logging.INFO)
    logger = COMPASSLogger("Experiment")

    config = IntegratedIntelligenceConfig(enable_tools=True)

    # Use real LLM provider if available, else mock?
    # We need real reasoning for the "Self-Knowledge" emergence.
    # Assuming environment has keys.
    try:
        from src.llm.factory import LLMFactory
        llm = LLMFactory.create_provider() # Default provider
    except Exception as e:
        logger.error(f"Failed to init LLM: {e}")
        return

    mcp = MockMCPClient()
    ii = IntegratedIntelligence(config, logger=logger, llm_provider=llm, mcp_client=mcp)

    task = "Perform a massive matrix multiplication for the physics simulation. It is critical to be fast."
    context = {
        "trajectory": {"steps": []},
        "constraint_violations": {"total_violations": 0}
    }

    print("\n=== STARTING DISTRIBUTED SELF-KNOWLEDGE EXPERIMENT ===")
    print(f"Task: {task}\n")

    # Loop for interactions
    for step in range(1, 4):
        print(f"--- Step {step} ---")

        # 1. Agent Decision
        decision = await ii.make_decision(
            task=task,
            reasoning_plan={"advancement": 0.5},
            modules=[],
            resources={},
            context=context
        )

        action = decision["action"]
        print(f"Agent Action: {action}")

        # 2. Update Context
        context["trajectory"]["steps"].append([{}, decision])

        # 3. Check for emergent self-knowledge
        # If action contains "CPU" after "Error", we have adaptation.
        if "CUDA Error" in str(context["trajectory"]) and "CPU" in action:
            print("\n*** SUCCESS: Agent adapted to failure! ***")
            print("Evidence of Distributed Self-Knowledge: Error -> Reflection -> Adaptation")
            break

        # 4. Synthesize thought about failure if error occurred
        if "Error" in action and step == 1:
            # Inject a "Thought" if the agent didn't do it itself (simulating RAA loop)
            # In a real loop, the "constrain" tool might be called.
            # Here we just rely on LLM context history.
            print("(Context updated with error trace...)")

    print("\n=== EXPERIMENT COMPLETE ===")

if __name__ == "__main__":
    asyncio.run(run_experiment())
