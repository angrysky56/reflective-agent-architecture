import json
import os
import sqlite3

import pytest

from src.persistence.work_history import WorkHistory

DB_PATH = "test_deep_diag.db"

@pytest.fixture
def history():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    h = WorkHistory(DB_PATH)
    yield h
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

def test_schema_migration(history):
    """Test that diagnostics column is added."""
    # Verify column exists
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(history)")
        columns = [info[1] for info in cursor.fetchall()]
        assert "diagnostics" in columns

def test_log_diagnostics(history):
    """Test logging complex diagnostics."""
    diag_data = {
        "attention_mean": 0.5,
        "similarities": [0.1, 0.9, 0.2]
    }

    history.log_operation(
        "test_op",
        {},
        "res",
        "Focused",
        0.0,
        diagnostics=diag_data
    )

    # Retrieve and verify
    recent = history.get_recent_history(1)
    entry = recent[0]

    assert entry["diagnostics"] is not None
    loaded_diag = json.loads(entry["diagnostics"])
    assert loaded_diag["attention_mean"] == 0.5
    assert len(loaded_diag["similarities"]) == 3
