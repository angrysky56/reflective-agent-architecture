from unittest.mock import MagicMock, patch

import pytest

from src.integration.sleep_cycle import SleepCycle


@pytest.fixture
def mock_workspace():
    ws = MagicMock()
    ws.config = MagicMock()
    ws.config.ruminator_enabled = True
    ws.config.ruminator_delay = 0.0
    ws.neo4j_driver = MagicMock()
    ws.collection = MagicMock()
    ws.ruminator_provider = MagicMock()
    return ws

@patch("time.sleep")
def test_ruminator_graph_connections(mock_sleep, mock_workspace):
    """Verify Ruminator finds and connects lonely nodes."""
    sc = SleepCycle(workspace=mock_workspace)

    # 1. Mock lonely nodes
    mock_session = MagicMock()
    mock_workspace.neo4j_driver.session.return_value.__enter__.return_value = mock_session

    # First run: Find lonely nodes
    # Second run: Check existence (mocking 'False' - not connected)
    # Third run: Create relationship

    # We need to control the return values of session.run carefully
    # The first call is for finding lonely nodes
    lonely_node = {"id": "node1", "name": "Concept A", "content": "Content A"}

    # The second call is for checking existence
    exists_result = {"connected": False}

    # The third call is for MERGE (returns nothing usually)

    def side_effect(*args, **kwargs):
        query = args[0].strip()
        if "size((n)--()) < 2" in query:
            return [lonely_node]
        if "exists((a)--(b))" in query:
            mock_result = MagicMock()
            mock_result.single.return_value = exists_result
            return mock_result
        return []

    mock_session.run.side_effect = side_effect

    # 2. Mock Chroma results
    mock_workspace.collection.query.return_value = {
        "ids": [["node2"]],
        "documents": [["Content B"]]
    }

    # 3. Mock Ruminator response
    mock_workspace.ruminator_provider.generate.return_value = "YES, RELATES_TO"

    # Execute
    result = sc._ruminate_on_graph_connections()

    # Verify
    assert result["status"] == "active"
    assert result["connections_made"] == 1

    # Verify MERGE called
    calls = mock_session.run.call_args_list
    merge_called = False
    for call in calls:
        query = call[0][0]
        if "MERGE (a)-[:RELATES_TO]->(b)" in query:
            merge_called = True
            break
    assert merge_called

def test_ruminator_graph_no_lonely_nodes(mock_workspace):
    """Verify idle status when no lonely nodes."""
    sc = SleepCycle(workspace=mock_workspace)

    mock_session = MagicMock()
    mock_workspace.neo4j_driver.session.return_value.__enter__.return_value = mock_session
    mock_session.run.return_value = [] # No nodes

    result = sc._ruminate_on_graph_connections()

    assert result["status"] == "idle"
    assert "No lonely nodes" in result["message"]
