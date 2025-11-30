import numpy as np
import pytest

from src.cognition.stereoscopic_engine import StereoscopicEngine


def test_stereoscopic_engine_initialization():
    """Test that the engine initializes its components correctly."""
    engine = StereoscopicEngine(embedding_dim=10)
    assert engine.embedding_dim == 10
    assert engine.continuity_field is not None
    assert engine.plasticity_gate is not None
    assert np.all(engine.current_state == np.zeros(10))

def test_intervention_flow():
    """
    Test the full flow of an intervention:
    1. Initialization
    2. Proposal (Intervention)
    3. Plasticity Gate Evaluation
    4. Continuity Field Integration
    """
    engine = StereoscopicEngine(embedding_dim=5)
    initial_state = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    engine.initialize_state(initial_state)

    # 1. Valid Intervention (Small change, low uncertainty)
    # Should be permitted by Plasticity Gate (Exploration mode)
    intervention = np.array([0.01, 0.01, 0.0, 0.0, 0.0])
    success, score, msg = engine.process_intervention(intervention, context="Test Valid Intervention")

    assert success is True
    assert "accepted" in msg
    assert np.allclose(engine.current_state, initial_state + intervention)

    # Check metrics
    metrics = engine.get_identity_metrics()
    assert metrics["anchors"] == 2  # Initial + 1 intervention
    assert metrics["modifications"] == 1

def test_plasticity_gate_rejection():
    """Test that the Plasticity Gate rejects incoherent interventions."""
    engine = StereoscopicEngine(embedding_dim=5)
    initial_state = np.zeros(5)
    engine.initialize_state(initial_state)

    # 2. Invalid Intervention (Massive change, high drift)
    # Should be rejected because it breaks coherence (Lipschitz constraint)
    # ContinuityField.validate_coherence will return low score for high drift/jump
    intervention = np.array([10.0, 10.0, 10.0, 10.0, 10.0])

    success, score, msg = engine.process_intervention(intervention, context="Test Invalid Intervention")

    assert success is False
    assert "rejected" in msg
    assert np.allclose(engine.current_state, initial_state) # State should not change

    metrics = engine.get_identity_metrics()
    assert metrics["anchors"] == 1 # Only initial
    assert metrics["modifications"] == 0 # No history recorded for rejection (or maybe it should? implementation says only if permitted)

def test_continuity_field_topology():
    """Test that the Continuity Field correctly calculates drift."""
    engine = StereoscopicEngine(embedding_dim=5)
    initial_state = np.zeros(5)
    engine.initialize_state(initial_state)

    # Add some anchors to define a manifold
    engine.continuity_field.add_anchor(np.array([0.1, 0.0, 0.0, 0.0, 0.0]))
    engine.continuity_field.add_anchor(np.array([0.2, 0.0, 0.0, 0.0, 0.0]))

    # A point ON the line defined by anchors should have low drift
    on_manifold = np.array([0.15, 0.0, 0.0, 0.0, 0.0])
    drift_low = engine.continuity_field.get_drift_metric(on_manifold)

    # A point FAR from the line should have high drift
    off_manifold = np.array([0.15, 1.0, 1.0, 0.0, 0.0])
    drift_high = engine.continuity_field.get_drift_metric(off_manifold)

    assert drift_low < drift_high
