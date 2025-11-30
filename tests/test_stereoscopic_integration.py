import asyncio
import unittest
from unittest.mock import MagicMock

from src.cognition.stereoscopic_engine import StereoscopicEngine
from src.compass.integrated_intelligence import IntegratedIntelligence, IntegratedIntelligenceConfig


class TestStereoscopicIntegration(unittest.TestCase):
    def setUp(self):
        self.config = IntegratedIntelligenceConfig(
            learning_rate=0.1,
            gamma_discounting=0.9
        )
        self.mock_engine = MagicMock(spec=StereoscopicEngine)
        self.mock_engine.embedding_dim = 4  # Small dim for testing

        self.intelligence = IntegratedIntelligence(
            config=self.config,
            stereoscopic_engine=self.mock_engine
        )

    def test_intervention_accepted(self):
        """Test that accepted intervention proceeds normally."""
        # Setup mock to accept
        self.mock_engine.process_intervention.return_value = (True, 0.9, "Accepted")

        # Run decision synthesis
        # We need to mock internal methods to avoid complex dependencies
        self.intelligence._uncertainty_intelligence = MagicMock(return_value=0.8)
        self.intelligence._evolutionary_intelligence = MagicMock(return_value=0.7)
        self.intelligence._neural_intelligence = MagicMock(return_value=0.6)
        self.intelligence._universal_intelligence = MagicMock(return_value=0.8)
        self.intelligence._generate_action = MagicMock(return_value="Test Action")
        self.intelligence._generate_reasoning = MagicMock(return_value="Test Reasoning")

        decision = asyncio.run(self.intelligence.make_decision(
            task="Test Task",
            reasoning_plan={},
            modules=[1, 2],
            resources={},
            context={}
        ))

        # Verify engine was called
        self.mock_engine.process_intervention.assert_called_once()

        # Verify score wasn't penalized (mock returns 0.8, so confidence should be 0.8)
        self.assertAlmostEqual(decision["confidence"], 0.8)
        self.assertEqual(decision["intelligence_breakdown"]["stereoscopic_gate"], 0.9)

    def test_intervention_rejected(self):
        """Test that rejected intervention penalizes score."""
        # Setup mock to reject
        self.mock_engine.process_intervention.return_value = (False, 0.2, "Rejected")

        # Run decision synthesis
        self.intelligence._uncertainty_intelligence = MagicMock(return_value=0.8)
        self.intelligence._evolutionary_intelligence = MagicMock(return_value=0.7)
        self.intelligence._neural_intelligence = MagicMock(return_value=0.6)
        self.intelligence._universal_intelligence = MagicMock(return_value=0.8)
        self.intelligence._generate_action = MagicMock(return_value="Test Action")
        self.intelligence._generate_reasoning = MagicMock(return_value="Test Reasoning")

        decision = asyncio.run(self.intelligence.make_decision(
            task="Test Task",
            reasoning_plan={},
            modules=[1, 2],
            resources={},
            context={}
        ))

        # Verify engine was called
        self.mock_engine.process_intervention.assert_called_once()

        # Verify score WAS penalized (0.8 * 0.5 = 0.4)
        self.assertAlmostEqual(decision["confidence"], 0.4)
        self.assertEqual(decision["intelligence_breakdown"]["stereoscopic_gate"], 0.2)

if __name__ == '__main__':
    unittest.main()
