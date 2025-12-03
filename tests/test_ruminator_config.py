from unittest.mock import MagicMock, patch

import pytest
from neo4j.exceptions import Neo4jError

from src.integration.sleep_cycle import SleepCycle


@pytest.fixture
def mock_workspace():
    ws = MagicMock()
    ws.config = MagicMock()
    ws.config.ruminator_enabled = True
    ws.config.ruminator_delay = 0.1
    ws.system_guide = MagicMock()
    ws.neo4j_driver = MagicMock()
    ws.ruminator_provider = MagicMock()
    return ws

def test_ruminator_disabled(mock_workspace):
    """Verify execution is skipped if disabled."""
    mock_workspace.config.ruminator_enabled = False
    sc = SleepCycle(workspace=mock_workspace)

    result = sc._ruminate_on_codebase()
    assert result["status"] == "skipped"
    assert "disabled" in result["reason"]

@patch("time.sleep")
def test_ruminator_delay(mock_sleep, mock_workspace):
    """Verify delay is respected."""
    sc = SleepCycle(workspace=mock_workspace)

    # Mock undocumented code found
    mock_session = MagicMock()
    mock_workspace.neo4j_driver.session.return_value.__enter__.return_value = mock_session
    mock_session.run.return_value = [{"id": "1", "snippet": "code", "file": "test.py", "line": 1}]

    # Mock successful generation
    mock_workspace.ruminator_provider.generate.return_value = "Docstring"

    sc._ruminate_on_codebase()

    # Verify sleep called with configured delay
    mock_sleep.assert_called_with(0.1)

def test_ruminator_db_error(mock_workspace):
    """Verify DB errors are caught."""
    sc = SleepCycle(workspace=mock_workspace)

    # Mock DB error
    mock_workspace.neo4j_driver.session.side_effect = Exception("DB Connection Failed")

    result = sc._ruminate_on_codebase()
    assert result["status"] == "error"
    assert "DB Connection Failed" in result["reason"]

@patch("time.sleep")
def test_ruminator_malformed_response(mock_sleep, mock_workspace):
    """Verify malformed responses are skipped."""
    sc = SleepCycle(workspace=mock_workspace)

    # Mock undocumented code found
    mock_session = MagicMock()
    mock_workspace.neo4j_driver.session.return_value.__enter__.return_value = mock_session
    mock_session.run.return_value = [{"id": "1", "snippet": "code", "file": "test.py", "line": 1}]

    # Mock error response
    mock_workspace.ruminator_provider.generate.return_value = "Error: Model failed"

    result = sc._ruminate_on_codebase()

    # Verify update NOT called for this item
    # The second call to run (SET b.notes...) should not happen
    # run is called once for MATCH, and then 0 times for SET
    assert mock_session.run.call_count == 1
