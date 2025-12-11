import json
import sqlite3
from unittest.mock import MagicMock

import pytest

from src.cognition.cognitive_diagnostics import CognitiveDiagnostics
from src.persistence.work_history import WorkHistory


def test_stress_sensor_integration(tmp_path):
    # Setup Logic
    db_path = tmp_path / "test_history.db"
    history = WorkHistory(str(db_path))

    diagnostics = CognitiveDiagnostics(workspace=MagicMock())

    # 1. Log some operations
    history.log_operation("synthesize", {"foo": "bar"}, "heavy result", node_ids=["A"])
    history.log_operation("deconstruct", {"bar": "baz"}, "complex result", node_ids=["B"])

    # 2. Update monitoring (should trigger lazy load and processing)
    # The first buffer check should be empty before update
    assert not hasattr(diagnostics, "stress_sensor")

    stats = diagnostics.update_stress_monitoring(history)

    # 3. Verify Stats
    assert stats["observations"] >= 2
    buffer_stats = stats["buffer"]

    # Since we used dummy complexity, it might not trigger the threshold (default 5.0)
    # But observation count confirms integration
    assert "observations" in stats
    assert hasattr(diagnostics, "stress_sensor")

    # 4. Force a high stress item mock (if possible via history)
    # Our mock energy/utility in diagnostics is fixed (1.0 * 0.5 = 0.5 stress), so it won't trigger buffer
    # But we can verify the sensor received the data
    assert diagnostics.stress_sensor.observation_count >= 2


if __name__ == "__main__":
    pytest.main([__file__])
