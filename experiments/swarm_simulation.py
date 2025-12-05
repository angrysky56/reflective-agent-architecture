import asyncio
import logging
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

from src.compass.adapters import RAALLMProvider
from src.integration.agent_factory import AgentFactory
from src.integration.swarm_controller import SwarmController


async def run_swarm_simulation():
    print("--- Experiment C: Swarm Dynamics Integration Test ---")

    # 1. Setup Infrastructure
    print("Initializing RAA Core Stack...")
    provider = RAALLMProvider()

    # We can mock tool execution for this test since we just want to test consensus-forming
    # and we aren't giving the agents files to read really.
    async def mock_executor(tool_name, args):
        return f"Tool {tool_name} executed with {args}"

    factory = AgentFactory(llm_provider=provider, tool_executor=mock_executor)
    swarm = SwarmController(agent_factory=factory)

    # 2. Define Problem (Multi-modal)
    # A problem that requires both Linear and Periodic thinking.
    context = (
        "Data Stream: [10, 15, 12, 17, 14, 19, 16, 21]\n"
        "Observation: The trend is generally going up, but it oscillates up and down."
    )
    task = "Predict the next 3 numbers. Explain the underlying pattern."

    print(f"\nTask: {task}")
    print(f"Context: {context}")

    # 3. Summon the Swarm
    advisors = ["linearist", "periodicist", "evolutionist"]
    print(f"\nSummoning Advisors: {advisors}...")

    # We need to peek at the swarm internals or just rely on the synthesis.
    # To debug why they missed the data, let's just inspect the final synthesis carefully.
    synthesis = await swarm.run_swarm(task, advisors, context)

    print("\n--- Hive Mind Consensus ---")
    print(synthesis)

    # 4. Validation
    # Check for actual insight, not just names
    success = False
    if "trend" in synthesis.lower() and "oscillat" in synthesis.lower() and "failure" not in synthesis.lower():
         print("\n[SUCCESS] Unified pattern detected (Trend + Oscillation).")
         success = True
    else:
         print("\n[FAILURE] Swarm failed to unify or detect pattern.")
         if "failure" in synthesis.lower():
             print("Reason: Swarm reported failure.")

if __name__ == "__main__":
    asyncio.run(run_swarm_simulation())
