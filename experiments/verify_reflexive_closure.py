import asyncio
import logging
import sys
import time
from unittest.mock import MagicMock, patch

import torch

# Add src to path
sys.path.append(".")

from src.director.director_core import DirectorConfig, DirectorMVP
from src.director.meta_pattern_analyzer import MetaPatternAnalyzer
from src.director.reflexive_closure_engine import ReflexiveClosureEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockManifold:
    def __init__(self):
        self.beta = 1.0

    def compute_adaptive_beta(self, entropy, max_entropy=None):
        return 0.5

    def set_beta(self, beta):
        self.beta = beta

async def run_simulation():
    logger.info("Starting Reflexive Closure Verification Simulation...")

    # Setup Director with Reflexive Closure
    config = DirectorConfig(
        enable_reflexive_closure=True,
        reflexive_analysis_interval=5, # Analyze every 5 interventions for fast feedback
        entropy_history_size=10
    )

    manifold = MockManifold()

    # Mock dependencies
    with patch('src.director.director_core.EntropyMonitor') as MockMonitor, \
         patch('src.director.director_core.SheafAnalyzer') as MockSheaf, \
         patch('src.director.director_core.MatrixMonitor') as MockMatrix, \
         patch('src.director.director_core.LTNRefiner') as MockLTN, \
         patch('src.director.director_core.HybridSearchStrategy') as MockHybrid, \
         patch('src.director.director_core.COMPASS') as MockCOMPASS, \
         patch('src.integration.agent_factory.AgentFactory') as MockAgentFactory:

        # Configure mocks
        mock_compass = MockCOMPASS.return_value
        mock_compass.omcd_controller.determine_resource_allocation.return_value = {"amount": 50.0, "confidence": 0.8}

        director = DirectorMVP(manifold=manifold, config=config)

        # Inject custom analyzer with low min_samples for testing
        director.reflexive_engine.analyzer = MetaPatternAnalyzer(min_samples=5)
        # Increase exploration rate to ensure we get variance in thresholds
        director.reflexive_engine.exploration_rate = 0.5

        # Ensure engine is initialized
        if not director.reflexive_engine:
            logger.error("Reflexive Engine not initialized!")
            return

        logger.info(f"Initial Threshold: {director.reflexive_engine.get_threshold('Unknown'):.3f}")

        # Simulation Loop
        # Scenario: System is struggling. High entropy.
        # We want to show that if interventions FAIL, the threshold might go UP (to intervene less)
        # Or if they SUCCEED, it might go DOWN (to intervene more).

        # Let's simulate: High Entropy -> Intervention -> SUCCESS
        # This should encourage the system to maintain or lower threshold (be more sensitive).

        logger.info("\n--- Phase 1: Successful Interventions (Should encourage sensitivity) ---")
        for i in range(15):
            # 1. Mock Monitor to report High Entropy (Clash)
            director.monitor.check_logits.return_value = (True, 2.5) # Entropy 2.5
            director.monitor.get_threshold.return_value = 2.0

            # 2. Mock Search to be SUCCESSFUL
            mock_result = MagicMock()
            mock_result.best_pattern = torch.tensor([1.0])
            mock_result.selection_score = 0.95 # High quality
            director.search = MagicMock(return_value=mock_result)

            # 3. Run Step
            logger.info(f"Step {i+1}: Triggering intervention...")
            director.check_and_search(
                current_state=torch.tensor([0.0]),
                processor_logits=torch.tensor([0.1, 0.9])
            )

            # Check threshold
            current_thresh = director.reflexive_engine.get_threshold('Unknown')
            logger.info(f"  -> Current Threshold: {current_thresh:.3f}")

            # Sleep slightly to ensure timestamps differ
            time.sleep(0.01)

        logger.info("\n--- Phase 2: Failed Interventions (Should reduce sensitivity) ---")
        # Now simulate failures. Interventions happen but result is poor.
        # The system should learn that intervening is useless and raise the threshold.
        for i in range(15):
            # 1. Mock Monitor to report High Entropy
            director.monitor.check_logits.return_value = (True, 2.5)

            # 2. Mock Search to be FAILURE (or low quality)
            mock_result = MagicMock()
            mock_result.best_pattern = torch.tensor([0.0])
            mock_result.selection_score = 0.1 # Low quality
            director.search = MagicMock(return_value=mock_result)

            # 3. Run Step
            logger.info(f"Step {i+16}: Triggering intervention (Failure)...")
            director.check_and_search(
                current_state=torch.tensor([0.0]),
                processor_logits=torch.tensor([0.1, 0.9])
            )

            # Check threshold
            current_thresh = director.reflexive_engine.get_threshold('Unknown')
            logger.info(f"  -> Current Threshold: {current_thresh:.3f}")
            time.sleep(0.01)

    logger.info("\nSimulation Complete.")

if __name__ == "__main__":
    asyncio.run(run_simulation())
