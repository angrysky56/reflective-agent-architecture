
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

    # Verify History
    logger.info("Verifying Work History...")
    history = workspace.history.get_recent_history(limit=20)

    found_deconstruct_cost = False
    found_hypothesize_cost = False

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

            if op_name == "deconstruct" and cost == "5.00":
                found_deconstruct_cost = True
            if op_name == "hypothesize" and cost == "1.00":
                found_hypothesize_cost = True

    if found_deconstruct_cost:
        logger.info("SUCCESS: Deconstruct cost recorded.")
    else:
        logger.error("FAILURE: Deconstruct cost NOT found.")

    if found_hypothesize_cost:
        logger.info("SUCCESS: Hypothesize cost recorded.")
    else:
        logger.error("FAILURE: Hypothesize cost NOT found.")

if __name__ == "__main__":
    asyncio.run(verify_costs())
