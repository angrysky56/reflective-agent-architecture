
import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.server import get_raa_context

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def verify_compass_tools():
    logger.info("Initializing RAA Context...")
    ctx = get_raa_context()

    # Ensure external MCP is initialized (mock or real)
    # For this test, we just need the internal tools

    logger.info("Getting Director...")
    director = ctx.get_director()

    if not director.compass:
        logger.error("COMPASS not initialized in Director")
        return

    logger.info("Asking COMPASS to inspect a node...")
    # We need a node ID. Let's create one first or use a dummy.
    # Actually, let's just ask it to "inspect the knowledge graph for node 'test_node'"
    # COMPASS should try to call 'inspect_knowledge_graph'

    # We can mock the tool execution or just see if it fails with "Tool not found"
    # But since we are running the real server context, it will try to execute.
    # 'inspect_knowledge_graph' calls workspace.get_node_context.

    task = "Inspect the knowledge graph for node 'test_node_123' to see its context."

    # We expect COMPASS to generate a plan and execute 'inspect_knowledge_graph'
    # We can't easily intercept the call unless we mock, but we can check the result.
    # If the tool is missing, it would have failed before.

    try:
        result = await director.compass.process_task(task)
        logger.info(f"COMPASS Result: {result}")

        # Check if the result indicates success or at least an attempt
        # If the tool was missing, it might say "I don't have a tool for that" or crash.

    except Exception as e:
        logger.error(f"COMPASS failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(verify_compass_tools())
