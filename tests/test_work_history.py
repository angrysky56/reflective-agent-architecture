import os
import sqlite3

import pytest

from src.persistence.work_history import WorkHistory

DB_PATH = "test_history.db"

@pytest.fixture
def history():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    h = WorkHistory(DB_PATH)
    yield h

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

def test_log_and_retrieve(history):
    """Test logging an operation and retrieving it."""
    history.log_operation(
        operation="test_op",
        params={"p1": "v1"},
        result={"message": "success"},
        cognitive_state="Focused",
        energy=-0.5
    )

    recent = history.get_recent_history(limit=1)
    assert len(recent) == 1
    entry = recent[0]

    assert entry["operation"] == "test_op"
    assert entry["cognitive_state"] == "Focused"
    assert entry["energy"] == -0.5
    assert "success" in entry["result_summary"]

def test_session_summary(history):
    """Test session summary stats."""
    history.log_operation("op1", {}, "res1", "Focused", -0.1)
    history.log_operation("op2", {}, "res2", "Focused", -0.2)
    history.log_operation("op3", {}, "res3", "Looping", -0.3)

    summary = history.get_session_summary()

    assert summary["total_operations"] == 3
    assert summary["dominant_state"] == "Focused"
