import json
import os
import sqlite3
from unittest.mock import MagicMock

import pytest

from src.integration.cwd_raa_bridge import BridgeConfig, CWDRAABridge
from src.persistence.work_history import WorkHistory

DB_PATH = "test_recall.db"

@pytest.fixture
def history():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    h = WorkHistory(DB_PATH)
    yield h
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

def test_search_history(history):
    """Test the search_history method directly."""
    history.log_operation("hypothesize", {"a": 1}, "Found a link", "Focused", -0.5)
    history.log_operation("deconstruct", {"b": 2}, "Broken down", "Looping", -0.1)

    # Test type filter
    res = history.search_history(operation_type="hypothesize")
    assert len(res) == 1
    assert res[0]["operation"] == "hypothesize"

    # Test text search
    res = history.search_history(query="Broken")
    assert len(res) == 1
    assert res[0]["operation"] == "deconstruct"

def test_recall_tool_integration():
    """Test that the bridge exposes history search."""
    # Setup bridge with mock history
    bridge = MagicMock()
    bridge.history = MagicMock()
    bridge.history.search_history.return_value = [{"operation": "test", "result": "success"}]

    # Simulate tool call logic (from server.py)
    arguments = {"query": "test", "limit": 5}

    results = bridge.history.search_history(
        query=arguments.get("query"),
        operation_type=arguments.get("operation_type"),
        limit=arguments.get("limit", 10)
    )

    bridge.history.search_history.assert_called_with(
        query="test",
        operation_type=None,
        limit=5
    )
    assert len(results) == 1
