
import asyncio
import logging
import os
import sys
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Any, Dict, List

from src.compass.adapters import RAALLMProvider
from src.director.allostatic_controller import AllostaticConfig
from src.director.director_core import DirectorConfig, DirectorMVP, Intervention
from src.integration.agent_factory import AgentFactory
from src.integration.swarm_controller import SwarmController

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AllostaticSim")

# Mock Manifold for entropy calculation
class MockManifold:
    def __init__(self):
        self.beta = 10.0
        self.stored_patterns = []

# Mock Processor for entropy injection
class MockProcessor:
    pass

# Mock Continuity Service
class MockContinuity:
    pass

async def main():
    logger.info("Starting Allostatic ECC Simulation...")

    # 1. Setup Director with Allostatic capabilities
    config = DirectorConfig()
    # Mocking dependencies
    manifold = MockManifold()

    # Use Real LLM Provider for Swarm interactions
    llm_provider = RAALLMProvider(model_name="google/gemini-2.0-flash-exp:free")

    # Mock tool executor for AgentFactory
    async def mock_tool_executor(name, args):
        logger.info(f"[MockTool] Executing {name} with {args}")
        return {"status": "success", "result": "mock_result"}

    agent_factory = AgentFactory(llm_provider, tool_executor=mock_tool_executor)

    # Initialize Director (which in turn initializes Swarm and Allostatic controllers)
    # We need to hack the initialization slightly since DirectorMVP creates its own instances
    # But we can replace them or rely on default init
    director = DirectorMVP(
        manifold=manifold,
        config=config,
        mcp_client=None, # Not needed for this specific test
        llm_provider=llm_provider,
        continuity_service=MockContinuity()
    )

    # Manually inject our pre-configured components if needed, or configure them
    # DirectorMVP init already created self.allostatic_controller
    # Let's adjust safety thresholds to make the test faster
    director.allostatic_controller.config.gradient_threshold = 0.1 # Sensitive trigger
    director.allostatic_controller.config.critical_threshold = 3.0 # Crash line

    logger.info(f"Allostatic Controller Config: History={director.allostatic_controller.config.history_window}, "
                f"Horizon={director.allostatic_controller.config.prediction_horizon}")

    # 2. Simulation Loop: Linear Entropy Increase (Simulating Signal Decay)
    # We will manually feed entropy values to the director's monitor
    entropy_stream = [1.0, 1.2, 1.5, 1.9, 2.4, 2.8, 3.2, 3.5]
    # Gradient: ~0.3-0.5 per step.
    # Step 4 (1.9) -> Step 5 (2.4): 2.4 + 3*slope -> predicted > 3.0?

    simulation_log = []

    for t, ent in enumerate(entropy_stream):
        logger.info(f"\n--- Time Step {t}: Measured Entropy = {ent:.2f} ---")

        # 1. Update Allostatic Controller History
        director.allostatic_controller.record_entropy(ent)

        # 2. Trigger Director State Check (which includes Allostatic Trigger Check)
        intervention = await director.check_proactive_interventions()

        # Log status
        pred = director.allostatic_controller.state.predicted_entropy
        status = {
            "step": t,
            "measured_entropy": ent,
            "predicted_entropy": pred,
            "intervention": intervention.type if intervention else "None"
        }
        simulation_log.append(status)

        if intervention and intervention.type == "allostatic_correction":
            logger.info(f"!!! INTERVENTION TRIGGERED: {intervention.type} from {intervention.source} !!!")
            logger.info(f"Content: {intervention.content}")
            logger.info("[SUCCESS] Proactive ECC deployed before crash.")
            break
        elif intervention:
             logger.info(f"Intervention: {intervention.type} (Not Allostatic)")
        else:
            logger.info(f"System stable. Predicted: {pred if pred else 'N/A'}")

        time.sleep(1) # Simulate time passing

    # 3. Verification
    # Check if we intervened BEFORE entropy hit 3.0 (Critical Threshold)
    intervened = any(x["intervention"] == "allostatic_correction" for x in simulation_log)
    if intervened:
        trigger_step = next(x for x in simulation_log if x["intervention"] == "allostatic_correction")
        logger.info(f"Intervention occurred at Step {trigger_step['step']} (Entropy={trigger_step['measured_entropy']})")

        if trigger_step["measured_entropy"] < 3.0:
            logger.info("PASS: Intervention was Proactive (Measured < Critical)")
        else:
            logger.warning("FAIL: Intervention was Reactive (Measured >= Critical)")
    else:
        logger.error("FAIL: No Intervention triggered.")

if __name__ == "__main__":
    asyncio.run(main())
