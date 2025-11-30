import numpy as np
import pytest

from src.integration.continuity_field import ContinuityField


def test_continuity_field_initialization():
    field = ContinuityField(embedding_dim=4, k_neighbors=3)
    assert field.embedding_dim == 4
    assert field.k_neighbors == 3
    assert len(field.anchors) == 0

def test_add_anchor():
    field = ContinuityField(embedding_dim=2)
    vec = np.array([1.0, 0.0])
    field.add_anchor(vec)
    assert len(field.anchors) == 1
    assert np.array_equal(field.anchors[0], vec)

def test_drift_metric_on_manifold():
    # Create a simple linear manifold: y = x
    field = ContinuityField(embedding_dim=2, k_neighbors=2)
    field.add_anchor(np.array([0.0, 0.0]))
    field.add_anchor(np.array([1.0, 1.0]))
    field.add_anchor(np.array([2.0, 2.0]))

    # Point exactly on the line should have near-zero drift
    query = np.array([1.5, 1.5])
    drift = field.get_drift_metric(query)
    assert drift < 1e-6

def test_drift_metric_off_manifold():
    # Create a simple linear manifold: y = x
    field = ContinuityField(embedding_dim=2, k_neighbors=2)
    field.add_anchor(np.array([0.0, 0.0]))
    field.add_anchor(np.array([1.0, 1.0]))
    field.add_anchor(np.array([2.0, 2.0]))

    # Point off the line (e.g., [1.5, 0.5])
    # The projection should be roughly [1.0, 1.0] (closest point on line y=x)
    # Distance is sqrt(0.5^2 + 0.5^2) = sqrt(0.5) approx 0.707
    query = np.array([1.5, 0.5])
    drift = field.get_drift_metric(query)
    assert drift > 0.1

def test_insufficient_anchors():
    field = ContinuityField(embedding_dim=2, k_neighbors=5)
    field.add_anchor(np.array([0.0, 0.0]))
    # Should not crash, but fallback to simple distance or mean
    query = np.array([1.0, 1.0])
    drift = field.get_drift_metric(query)
    assert drift >= 0.0
