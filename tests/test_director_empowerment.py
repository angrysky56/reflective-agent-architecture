import asyncio
import os
import sys
from pathlib import Path

import tempfile
import shutil

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_director_empowerment")

# Use temp dir for Chroma to avoid crashes/corruption during test
temp_chroma_dir = tempfile.mkdtemp()
os.environ["CHROMA_PATH"] = temp_chroma_dir

try:
    from src.server import RAAServerContext
except Exception as e:
    logger.error(f"Failed to import RAAServerContext: {e}")
    shutil.rmtree(temp_chroma_dir)
    sys.exit(1)


async def mock_process_task(task, context=None):
    logger.info("MOCK COMPASS RECEIVED TASK")
    logger.info(f"Task: {task}")
    logger.info(f"Context: {context}")
    return {"status": "success", "context_received": context}


async def mock_call_tool(name, args):
    if name == "mcp_check_cognitive_state":
        logger.info("Mocking mcp_check_cognitive_state -> Looping")
        return "State: Looping (Stability: 0.1) - Stuck in recursive introspection."
    return "Unknown Tool"


async def main():
    logger.info("Initializing RAA Server Context...")
    ctx = RAAServerContext()
    ctx.initialize()

    director = ctx.workspace.director
    # It might be wrapped in SubstrateAwareDirector
    if hasattr(director, "director"):
        logger.info("Unwrapping SubstrateAwareDirector...")
        director = director.director

    logger.info(f"Director type: {type(director)}")

    # 1. Verify Pointer Injection
    if hasattr(director, "goal_controller") and director.goal_controller:
        logger.info("SUCCESS: Director has GoalController!")
    else:
        logger.error("FAILURE: Director missing GoalController!")
        return

    # 2. Mock Components
    # Mock COMPASS.process_task to capture what the Director sends
    director.compass.process_task = mock_process_task

    # Mock MCP Client to simulate "Looping" state
    director.mcp_client.call_tool = mock_call_tool

    # 3. Trigger Test
    # We pass a context with "trajectory" containing "substrate_transaction"
    # because the logic requires:
    # if cognitive_state_label == "Looping/Stuck" and "substrate_transaction" in str(context.get("trajectory", "")):

    test_context = {
        "trajectory": "Step 1: internal_monologue\nStep 2: substrate_transaction (cost 0.5)\nStep 3: internal_monologue",
        "force_time_gate": False,
    }

    logger.info("Running Director process_task_with_time_gate...")
    result = await director.process_task_with_time_gate(
        "I am trying to solve the problem but I keep thinking about how I am thinking.",
        context=test_context,
    )

    # 4. Verify Results
    received_context = result.get("context_received", {})
    prescription = received_context.get("prescription", "")
    exec_state = received_context.get("executive_state", {})

    logger.info("-" * 50)
    logger.info(f"Prescription: {prescription}")
    logger.info(f"Executive State: {exec_state}")
    logger.info("-" * 50)

    if "inspect_graph" in prescription:
        logger.info("SUCCESS: Director prescribed 'inspect_graph' for Hypochondria!")
    else:
        logger.error(f"FAILURE: Expected 'inspect_graph' in prescription, got: '{prescription}'")

    if exec_state.get("cognitive_label") == "Looping/Stuck":
        logger.info("SUCCESS: Cognitive State correctly identified as Looping/Stuck")
    else:
        logger.error("FAILURE: Cognitive State not correctly identified.")


if __name__ == "__main__":
    asyncio.run(main())
