import asyncio
import logging
import math
import os
import sys

import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cognition.meta_validator import MetaValidator


async def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("repro_nan_error")

    logger.info("Starting reproduction script for NaN error in MetaValidator...")

    validator = MetaValidator()

    # Case 1: Insight with NaN metrics
    logger.info("Case 1: Insight with NaN metrics")
    insight_nan = {
        "statement": "This is a NaN statement",
        "coherence": float("nan"),
        "resonance": float("nan"),
        "novelty": float("nan"),
    }

    try:
        score = validator.calculate_unified_score(
            insight_nan
        )  # Assuming this method calculates score from dict
        logger.info(f"Score for NaN insight: {score}")
    except Exception as e:
        logger.error(f"Error in Case 1: {e}")

    # Case 2: Insight with Inf metrics
    logger.info("Case 2: Insight with Inf metrics")
    insight_inf = {
        "statement": "This is an Inf statement",
        "coherence": float("inf"),
        "resonance": float("inf"),
        "novelty": float("inf"),
    }

    try:
        # Assuming we can invoke validation/scoring directly
        # We need to see how MetaValidator is used.
        # Based on previous view, it has `calculate_unified_score(c, r)`

        c_nan = float("nan")
        r_nan = float("nan")
        score = validator.calculate_unified_score(c_nan, r_nan)
        logger.info(f"Score for NaN/NaN input: {score}")

    except Exception as e:
        logger.error(f"Error in Case 2: {e}")


if __name__ == "__main__":
    asyncio.run(main())
