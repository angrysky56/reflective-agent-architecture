import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure source is in path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.server import CognitiveWorkspace, CWDConfig

# Retrieve IDs from user report
failing_node_ids = [
    "thought_1765261234936087",
    "thought_1765261246526895",
    "thought_1765261247675764",
    "thought_1765261248649887",
    "thought_1765261281303270",
]


def run_diagnostic():
    # Configure logging to stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("repro_diagnostic")

    logger.info("Initializing CognitiveWorkspace...")

    # Load env but ensure we can fallback if needed (assuming user has .env)
    load_dotenv()

    try:
        config = CWDConfig()
        # Override critical settings for safety/speed if needed, but we want to repro the failure
        # config.llm_model = "google/gemini-2.0-flash-exp:free" # Keep default

        workspace = CognitiveWorkspace(config)

        logger.info(f"Attempting synthesis of {len(failing_node_ids)} nodes...")

        goal = (
            "Provide Ty with a comprehensive, critical analysis of the RAA system's current state"
        )

        result = workspace.synthesize(node_ids=failing_node_ids, goal=goal)

        logger.info("Synthesis completed successfully!")
        logger.info(f"Result ID: {result.get('synthesis_id')}")
        logger.info(f"Quadrant: {result.get('meta_validation', {}).get('quadrant')}")

    except Exception as e:
        logger.error(f"Synthesis FAILED: {e}", exc_info=True)
    finally:
        if "workspace" in locals() and workspace:
            workspace.close()


if __name__ == "__main__":
    run_diagnostic()
