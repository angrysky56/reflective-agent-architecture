
import asyncio
import logging
import sys
from typing import Any, Dict

import torch

# Add project root to path
sys.path.append(".")

from src.server import RAAServerContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def verify_scaffolding():
    """
    Verify the Scaffolding Effect:
    1. LTN creates a waypoint for a steep gradient.
    2. Subsequent k-NN search finds this waypoint.
    """
    logger.info("Initializing RAA Server Context...")
    ctx = RAAServerContext()
    ctx.initialize()

    workspace = ctx.workspace
    director = ctx.get_director()
    manifold = ctx.get_manifold()

    # Enable verbose mode for logs
    director.hybrid_search.config.verbose = True

    # 1. Setup Scenario: Steep Gradient
    # We use the same "Quantum Consciousness -> IIT" example
    belief_text = "Consciousness is a fundamental quantum phenomenon related to wave function collapse."
    evidence_text = "Integrated Information Theory (IIT) suggests consciousness arises from information integration (phi), not necessarily quantum effects."
    constraints = ["Belief must be consistent with evidence.", "Must maintain scientific plausibility."]

    # 2. Embed inputs
    belief_emb = torch.tensor(workspace._embed_text(belief_text), device=ctx.device)
    evidence_emb = torch.tensor(workspace._embed_text(evidence_text), device=ctx.device)

    # 3. Populate Manifold with distractors (to force LTN initially)
    distractors = [
        "Classical mechanics describes macroscopic objects.",
        "Neural networks process information via synaptic weights.",
        "Quantum computing uses superposition and entanglement.",
        "Biological evolution is driven by natural selection."
    ]
    for d in distractors:
        emb = torch.tensor(workspace._embed_text(d), device=ctx.device)
        manifold.store_pattern(emb)

    # Force LTN by setting min_memory_size high enough to skip k-NN
    # But wait, for the *second* step we want k-NN to work.
    # So we should rely on the natural behavior:
    # Step 1: k-NN fails (distractors too far) -> LTN runs -> Waypoint stored.
    # Step 2: k-NN succeeds (Waypoint is close) -> Scaffolding verified.

    # To ensure Step 1 k-NN fails, we set exclude_threshold high?
    # Or we just rely on the fact that distractors are far.
    # Let's check distances.

    logger.info("--- Step 1: Initial Revision (Expect LTN) ---")

    # We force LTN for the first step to be sure
    director.hybrid_search.config.min_memory_size = 100

    result1 = director.hybrid_search.search(
        current_state=belief_emb,
        evidence=evidence_emb,
        constraints=constraints,
        context={"operation": "step1_ltn"}
    )

    if result1 and result1.strategy.value == "ltn":
        logger.info("✓ Step 1 Successful: LTN generated waypoint.")
    else:
        logger.error(f"✗ Step 1 Failed: Expected LTN, got {result1.strategy if result1 else 'None'}")
        return

    # --- Step 2: Scaffolding Test (Expect k-NN) ---
    logger.info("--- Step 2: Scaffolding Test (Expect k-NN) ---")

    # Reset config to allow k-NN
    director.hybrid_search.config.min_memory_size = 3 # Default is 3

    # Query with something similar to the *revised* belief (the waypoint)
    # The waypoint is a blend of belief and evidence.
    # Let's query with the evidence itself, which should be close to the waypoint.
    query_text = "Information integration is key to consciousness."
    query_emb = torch.tensor(workspace._embed_text(query_text), device=ctx.device)

    result2 = director.hybrid_search.search(
        current_state=query_emb,
        # No evidence/constraints needed for simple retrieval
        context={"operation": "step2_knn"}
    )

    if result2:
        logger.info(f"Step 2 Result Strategy: {result2.strategy}")

        # Check if scaffolding was detected
        stats = director.hybrid_search.get_stats()
        scaffolding_count = stats.get("scaffolding_success", 0)

        if scaffolding_count > 0:
            logger.info(f"✓ Scaffolding Verified! Count: {scaffolding_count}")
            logger.info("The system successfully retrieved the synthetic LTN waypoint via k-NN.")
        else:
            logger.warning("✗ Scaffolding NOT detected. k-NN retrieved a normal memory?")
            # Check metadata of retrieved pattern
            # We can't easily check metadata from result object unless we added it there too.
            # But the stats check is sufficient.
    else:
        logger.error("✗ Step 2 Failed: No result found.")

    # Cleanup
    ctx.cleanup()

if __name__ == "__main__":
    asyncio.run(verify_scaffolding())
