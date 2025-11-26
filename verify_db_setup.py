import os

from src.persistence.work_history import WorkHistory

DB_PATH = "verify_setup.db"

def test_db_creation():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    print(f"Initializing WorkHistory at {DB_PATH}...")
    history = WorkHistory(DB_PATH)

    if os.path.exists(DB_PATH):
        print("SUCCESS: Database file created.")
    else:
        print("FAILURE: Database file NOT created.")
        exit(1)

    # Verify schema
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(history)")
    columns = [info[1] for info in cursor.fetchall()]
    print(f"Columns: {columns}")

    if "diagnostics" in columns:
        print("SUCCESS: 'diagnostics' column present.")
    else:
        print("FAILURE: 'diagnostics' column missing.")
        exit(1)

    os.remove(DB_PATH)

if __name__ == "__main__":
    test_db_creation()
