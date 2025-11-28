
import asyncio
import logging
import sys
from typing import Any, Dict

import torch

# Add project root to path
sys.path.append(".")

from src.director.director_core import DirectorConfig
from src.director.ltn_refiner import LTNConfig
from src.server import RAAServerContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def reproduce_failure():
    """
    Reproduce the 'Quantum Consciousness -> IIT' revision failure.
    """
    logger.info("Initializing RAA Server Context...")
    ctx = RAAServerContext()
    ctx.initialize()

    workspace = ctx.workspace
    director = ctx.get_director()
    manifold = ctx.get_manifold()

    # 1. Setup Scenario
    belief_text = "Consciousness is a fundamental quantum phenomenon related to wave function collapse."
    evidence_text = "Integrated Information Theory (IIT) suggests consciousness arises from information integration (phi), not necessarily quantum effects."
    constraints = ["Belief must be consistent with evidence.", "Must maintain scientific plausibility."]

    logger.info(f"Belief: {belief_text}")
    logger.info(f"Evidence: {evidence_text}")

    # 2. Embed inputs
    belief_emb = torch.tensor(workspace._embed_text(belief_text), device=ctx.device)
    evidence_emb = torch.tensor(workspace._embed_text(evidence_text), device=ctx.device)

    # 3. Populate Manifold (optional - to simulate "sparse" but not empty memory)
    # If we leave it empty, it bootstraps. The user said "Manifold too sparse", implying some content.
    # Let's add some distractor patterns to create an energy landscape.
    distractors = [
        "Classical mechanics describes macroscopic objects.",
        "Neural networks process information via synaptic weights.",
        "Quantum computing uses superposition and entanglement.",
        "Biological evolution is driven by natural selection."
    ]
    for d in distractors:
        emb = torch.tensor(workspace._embed_text(d), device=ctx.device)
        manifold.store_pattern(emb)

    logger.info("Manifold populated with distractors.")

    # 4. Execute Hybrid Search (Revise)
    # We need to force LTN to reproduce the failure.
    # The default behavior tries k-NN first. If it finds a neighbor (even a distractor), it returns.
    # To test LTN, we can set knn_exclude_threshold to be very strict or modify the config.

    # Let's modify the director's hybrid config on the fly to force LTN
    # We can do this by setting min_memory_size to be larger than the current memory (4 patterns).

    director.hybrid_search.config.min_memory_size = 100

    logger.info("Executing Hybrid Search (Revise) with forced LTN (min_memory_size=100)...")
    result = director.hybrid_search.search(
        current_state=belief_emb,
        evidence=evidence_emb,
        constraints=constraints,
        context={"operation": "reproduce_failure"}
    )

    if result:
        logger.info("SUCCESS: Revision found a result.")
        logger.info(f"Strategy: {result.strategy}")
        logger.info(f"Selection Score: {result.selection_score}")
        logger.info(f"LTN Attempted: {result.ltn_attempted}")

        # Decode
        best_emb = result.best_pattern.cpu().tolist()
        query_result = workspace.collection.query(
            query_embeddings=[best_emb],
            n_results=1
        )
        if query_result["documents"] and query_result["documents"][0]:
            logger.info(f"Nearest Thought: {query_result['documents'][0][0]}")
        else:
            logger.info("Nearest Thought: None found in Chroma.")

    else:
        logger.error("FAILURE: Revision returned None.")

        # Inspect LTN Refiner stats if available
        logger.info(f"LTN Stats: {director.ltn_refiner.refinement_stats}")

    # Cleanup
    ctx.cleanup()

if __name__ == "__main__":
    asyncio.run(reproduce_failure())
