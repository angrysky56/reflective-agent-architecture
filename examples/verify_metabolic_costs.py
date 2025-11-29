
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.server import get_raa_context

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def verify_costs():
    logger.info("Initializing RAA Context...")
    ctx = get_raa_context()

    # Ensure initialized
    if not ctx.is_initialized:
        ctx.initialize()

    workspace = ctx.workspace
    if not workspace:
        logger.error("Workspace not initialized")
        return

    logger.info("Running Deconstruct (Cost: 5.0 J)...")
    from src.server import call_tool

    # Run Deconstruct
    try:
        await call_tool("deconstruct", {"problem": "The nature of entropy in cognitive systems"})
        logger.info("Deconstruct completed.")
    except Exception as e:
        logger.error(f"Deconstruct failed: {e}")

    logger.info("Running Hypothesize (Cost: 1.0 J)...")
    try:
        await call_tool("hypothesize", {
            "node_a_id": "dummy_1",
            "node_b_id": "dummy_2",
            "context": "Testing metabolic costs"
        })
        logger.info("Hypothesize completed (or attempted).")
    except Exception as e:
        logger.info(f"Hypothesize failed as expected (dummy IDs): {e}")

    # 3. Explore (Cost: 1.0 J)
    logger.info("Running Explore (Cost: 1.0 J)...")
    try:
        await ctx.call_tool("explore_for_utility", {"focus_area": "cognitive architecture"})
    except Exception as e:
        logger.warning(f"Explore failed with error: {e}")

    # Verify History
    logger.info("Verifying Work History...")
    history = workspace.history.get_recent_history(limit=50)

    found_deconstruct_cost = False
    found_hypothesize_cost = False
    found_explore_cost = False

    import json
    for entry in history:
        op = entry.get("operation")
        params_str = entry.get("params", "{}")
        try:
            params = json.loads(params_str) if isinstance(params_str, str) else params_str
        except json.JSONDecodeError:
            params = {}

        if op == "substrate_transaction":
            cost = params.get("cost")
            op_name = params.get("operation")
            logger.info(f"Found Transaction: {op_name} -> {cost} J")

            if op_name == "deconstruct" and "5.0" in str(cost):
                found_deconstruct_cost = True
            elif op_name == "hypothesize" and "1.0" in str(cost):
                found_hypothesize_cost = True
            elif op_name == "explore_for_utility" and "1.0" in str(cost):
                found_explore_cost = True

    if not found_deconstruct_cost:
        logger.error("FAILURE: Deconstruct cost NOT found.")
    if not found_hypothesize_cost:
        logger.error("FAILURE: Hypothesize cost NOT found.")
    if not found_explore_cost:
        logger.error("FAILURE: Explore cost NOT found.")

    if found_deconstruct_cost and found_hypothesize_cost and found_explore_cost:
        logger.info("SUCCESS: All metabolic costs verified!")
    else:
        logger.error("FAILURE: Not all metabolic costs were found.")

if __name__ == "__main__":
    asyncio.run(verify_costs())
