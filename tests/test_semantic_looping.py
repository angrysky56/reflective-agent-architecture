from unittest.mock import MagicMock

import pytest

from src.cognition.cognitive_diagnostics import CognitiveDiagnostics
from src.persistence.work_history import WorkHistory

# Tests for Operation Categorization are still valid if imported from proper place
from src.substrate.operation_categories import (
    OperationCategory,
    get_category,
    is_exempt_from_looping,
)


def test_operation_categories():
    assert get_category("substrate_transaction") == OperationCategory.INFRASTRUCTURE
    assert get_category("synthesize") == OperationCategory.DELIBERATION
    assert is_exempt_from_looping("substrate_transaction") is True
    assert is_exempt_from_looping("deconstruct") is False


def test_work_history_node_tracking(tmp_path):
    db_path = tmp_path / "test_history.db"
    history = WorkHistory(str(db_path))

    # Log ops with nodes
    history.log_operation("synthesize", {}, "res", node_ids=["node_A"])
    history.log_operation("synthesize", {}, "res", node_ids=["node_A"])
    history.log_operation("substrate_transaction", {}, "res", node_ids=["node_A"])  # Infrastructure

    # Check filtering
    delib = history.get_deliberation_history(limit=10)
    assert len(delib) == 2  # Should ignore substrate_transaction

    # Check stats
    stats = history.get_node_visitation_stats(limit=10)
    assert stats["node_A"] == 2


def test_cognitive_diagnostics_looping():
    # Mock workspace and history
    mock_workspace = MagicMock()
    # Mock neo4j driver session to return no flag (default)
    mock_session = MagicMock()
    mock_workspace.neo4j_driver.session.return_value.__enter__.return_value = mock_session
    mock_session.run.return_value.single.return_value = None  # No flag

    diagnostics = CognitiveDiagnostics(mock_workspace)
    mock_history = MagicMock()

    # Scenario A: Fixation (Single node 10 times)
    mock_history.get_node_visitation_stats.return_value = {"node_fixated": 10}
    is_looping, top_node, count, warnings = diagnostics.detect_semantic_looping(mock_history)
    assert is_looping is True
    assert top_node == "node_fixated"
    assert count == 10

    # Scenario B: Healthy (3 nodes, distributed)
    mock_history.get_node_visitation_stats.return_value = {"node_A": 3, "node_B": 3, "node_C": 3}
    is_looping, top_node, count, warnings = diagnostics.detect_semantic_looping(mock_history)
    assert is_looping is False


if __name__ == "__main__":
    pytest.main([__file__])
