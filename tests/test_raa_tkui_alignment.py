
import os
import shutil
import unittest
from pathlib import Path

import torch

from src.compass.compass_framework import COMPASS
from src.compass.meta_controller import MetaController, WorkflowType
from src.integration.continuity_service import ContinuityService
from src.integration.precuneus import PrecuneusIntegrator
from src.persistence.work_history import WorkHistory


class TestRAATKUIAlignment(unittest.TestCase):

    def setUp(self):
        # Setup temporary DB
        self.test_db = "test_raa_history.db"
        self.work_history = WorkHistory(self.test_db)
        self.continuity_service = ContinuityService(self.work_history)
        self.precuneus = PrecuneusIntegrator(dim=64)
        self.meta_controller = MetaController()

    def tearDown(self):
        # Clean up DB
        if os.path.exists(self.test_db):
            os.remove(self.test_db)

    def test_meta_controller_workflow_selection(self):
        """Test that MetaController selects the correct workflow."""
        self.assertEqual(self.meta_controller._determine_workflow("Fix the bug in server.py", None), WorkflowType.DEBUG)
        self.assertEqual(self.meta_controller._determine_workflow("Research quantum physics", None), WorkflowType.RESEARCH)
        self.assertEqual(self.meta_controller._determine_workflow("Write a poem about AI", None), WorkflowType.CREATIVE)
        self.assertEqual(self.meta_controller._determine_workflow("Calculate 2+2", None), WorkflowType.STANDARD)

    def test_continuity_service_signature(self):
        """Test calculation of causal signature."""
        # Log some history
        self.work_history.log_operation("test_op", {"agent_id": "Explorer"}, "success", causal_impact=0.8)
        self.work_history.log_operation("test_op", {"agent_id": "Explorer"}, "success", causal_impact=0.5)

        signature = self.continuity_service.get_causal_signature("Explorer", dim=64)

        self.assertEqual(signature.shape, (64,))
        self.assertTrue(torch.norm(signature) > 0)

        # Test empty history
        empty_sig = self.continuity_service.get_causal_signature("Ghost", dim=64)
        self.assertEqual(torch.norm(empty_sig), 0)

    def test_precuneus_continuity_field(self):
        """Test Precuneus forward pass with Causal Signature."""
        dim = 64
        vectors = {
            'state': torch.randn(dim),
            'agent': torch.randn(dim),
            'action': torch.randn(dim)
        }
        energies = {
            'state': torch.tensor(1.0),
            'agent': torch.tensor(1.0),
            'action': torch.tensor(1.0)
        }

        # Run without signature
        out_standard = self.precuneus(vectors, energies)

        # Run with signature
        signature = torch.randn(dim)
        out_continuity = self.precuneus(vectors, energies, causal_signature=signature)

        # Outputs should differ because weights are modulated
        self.assertFalse(torch.allclose(out_standard, out_continuity))

    def test_work_history_causal_impact(self):
        """Test logging and retrieving causal impact."""
        self.work_history.log_operation("impact_test", {}, "result", causal_impact=0.99)

        history = self.work_history.get_recent_history(limit=1)
        self.assertEqual(len(history), 1)
        self.assertAlmostEqual(history[0]['causal_impact'], 0.99)

if __name__ == '__main__':
    unittest.main()
