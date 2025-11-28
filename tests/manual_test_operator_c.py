
import asyncio
import logging
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.integration.cwd_raa_bridge import CWDRAABridge
from src.manifold import Manifold
from src.server import CognitiveWorkspace, RAAServerContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_operator_c():
    logger.info("Starting Operator C (Belief Revision) Test...")

    # Initialize Context (Mocking where necessary)
    # We need a real workspace connected to Neo4j and Ollama
    try:
        ctx = RAAServerContext()
        ctx.initialize()
        workspace = ctx.workspace

        if not workspace:
            logger.error("Failed to initialize workspace")
            return

        # 1. Create Initial Belief
        logger.info("Creating initial belief: 'All swans are white.'")
        with workspace.neo4j_driver.session() as session:
            belief_id = workspace._create_thought_node(
                session,
                "All swans are white.",
                "belief",
                confidence=0.9,
                embedding=workspace._embed_text("All swans are white.")
            )
            logger.info(f"Belief created: {belief_id}")

        # 2. Revise Belief
        evidence = "I saw a black swan in Australia."
        logger.info(f"Revising with evidence: '{evidence}'")

        result = workspace.revise(belief_id, evidence)

        if "error" in result:
            logger.error(f"Revision failed: {result['error']}")
            return

        # 3. Verify Result
        logger.info("Revision Result:")
        logger.info(f"Original ID: {result['original_id']}")
        logger.info(f"Revision ID: {result['revision_id']}")
        logger.info(f"Revised Content: {result['revision_content']}")
        logger.info(f"Loss: {result['loss']}")
        logger.info(f"Metrics: {result['metrics']}")

        # Check Content
        revised_content = result['revision_content'].lower()
        if "black" in revised_content and "white" in revised_content:
            logger.info("SUCCESS: Revision incorporates both concepts.")
        else:
            logger.warning("FAILURE: Revision might be missing concepts.")

        # Check Graph Link
        with workspace.neo4j_driver.session() as session:
            link = session.run(
                """
                MATCH (new:ThoughtNode {id: $new_id})-[r:REVISES]->(old:ThoughtNode {id: $old_id})
                RETURN r
                """,
                new_id=result['revision_id'],
                old_id=result['original_id']
            ).single()

            if link:
                logger.info("SUCCESS: REVISES relationship exists in graph.")
            else:
                logger.error("FAILURE: REVISES relationship missing.")

    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'ctx' in locals():
            ctx.cleanup()

if __name__ == "__main__":
    test_operator_c()
