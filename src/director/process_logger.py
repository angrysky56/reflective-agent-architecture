import json
import os
import sqlite3
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# Constants
# Use absolute path relative to this file to ensure it works from anywhere
# Correct path: src/director/ -> src/raa_history.db (one level up)
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../raa_history.db"))
# print(f"DEBUG: ProcessLogger using DB at {DB_PATH}")


class ProcessLogger:
    _instance = None

    def __new__(cls) -> "ProcessLogger":
        if cls._instance is None:
            cls._instance = super(ProcessLogger, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize the SQLite database."""
        try:
            self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            self.cursor = self.conn.cursor()
            self._create_table()
        except Exception as e:
            print(f"Error initializing ProcessLogger: {e}")

    def _create_table(self) -> None:
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS process_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                event_type TEXT,
                content TEXT,
                source TEXT
            )
        """
        )
        self.conn.commit()

    def log(self, event_type: str, content: Dict[str, Any], source: str = "Director") -> None:
        """Log an event to the database.

        event_type: "THOUGHT", "STATE", "SWARM", "ENERGY", "GOAL"
        """
        try:
            timestamp = time.time()
            # Ensure content is JSON serializable
            try:
                content_json = json.dumps(content)
            except TypeError:
                content_json = json.dumps(
                    {"error": "Content not serializable", "raw": str(content)}
                )

            self.cursor.execute(
                "INSERT INTO process_logs (timestamp, event_type, content, source) VALUES (?, ?, ?, ?)",
                (timestamp, event_type, content_json, source),
            )
            self.conn.commit()
            # print(f"LOGGED: {event_type} - {source}")
        except Exception as e:
            print(f"Error logging event: {e}")

    def get_recent_logs(
        self, limit: int = 100, event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve recent logs for the dashboard."""
        try:
            if event_type:
                self.cursor.execute(
                    "SELECT timestamp, event_type, content, source FROM process_logs WHERE event_type = ? ORDER BY timestamp DESC LIMIT ?",
                    (event_type, limit),
                )
            else:
                self.cursor.execute(
                    "SELECT timestamp, event_type, content, source FROM process_logs ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                )

            rows = self.cursor.fetchall()
            logs = []
            for row in rows:
                try:
                    content_data = json.loads(row[2])
                except json.JSONDecodeError:
                    content_data = {"raw": row[2]}

                logs.append(
                    {
                        "timestamp": row[0],
                        "datetime": datetime.fromtimestamp(row[0]).strftime("%H:%M:%S"),
                        "event_type": row[1],
                        "content": content_data,
                        "source": row[3],
                    }
                )
            return logs
        except Exception as e:
            print(f"Error reading logs: {e}")
            return []

    def get_energy_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Specific helper for the Energy Landscape."""
        return self.get_recent_logs(limit, "ENERGY")

    def get_latest_state(self) -> Dict[str, Any]:
        """Get the most recent Cognitive State log."""
        logs = self.get_recent_logs(1, "STATE")
        if logs:
            content = logs[0]["content"]
            if isinstance(content, dict):
                return content
        return {"state": "Unknown", "energy": 0.0, "stability": "Unknown"}


# Singleton Accessor
logger = ProcessLogger()
