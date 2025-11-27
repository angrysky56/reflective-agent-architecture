import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class WorkHistory:
    """
    Persists agent operations, results, and cognitive states to a SQLite database.
    Allows the agent to 'recall' its work history and conceptual location.
    """

    def __init__(self, db_path: str = "raa_history.db"):
        # Resolve to absolute path in src/ directory
        # This file is in src/persistence/, so .parent.parent is src/
        if db_path == "raa_history.db":
            self.db_path = str(Path(__file__).parent.parent / "raa_history.db")
        else:
            self.db_path = db_path

        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        operation TEXT NOT NULL,
                        params TEXT,
                        result_summary TEXT,
                        cognitive_state TEXT,
                        energy REAL
                    )
                """)

                # Check for diagnostics column and add if missing (Migration)
                cursor.execute("PRAGMA table_info(history)")
                columns = [info[1] for info in cursor.fetchall()]
                if "diagnostics" not in columns:
                    cursor.execute("ALTER TABLE history ADD COLUMN diagnostics TEXT")

                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize history DB: {e}")

    def log_operation(
        self,
        operation: str,
        params: Dict[str, Any],
        result: Any,
        cognitive_state: str = "Unknown",
        energy: float = 0.0,
        diagnostics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an operation and its context to history.
        """
        try:
            # Serialize params and result summary
            params_json = json.dumps(params, default=str)
            diagnostics_json = json.dumps(diagnostics, default=str) if diagnostics else None

            # Create a brief summary of the result
            if isinstance(result, dict):
                # If result has a 'message' or 'summary', use that
                summary = result.get("message") or result.get("summary") or str(result)[:200]
            else:
                summary = str(result)[:200]

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO history (operation, params, result_summary, cognitive_state, energy, diagnostics)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (operation, params_json, summary, cognitive_state, energy, diagnostics_json))
                conn.commit()

            logger.debug(f"Logged operation '{operation}' to history.")

        except Exception as e:
            logger.error(f"Failed to log operation to history: {e}")

    def get_recent_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve recent history entries.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM history
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))

                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve history: {e}")
            return []

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get aggregate statistics for the current session (or all time).
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Count total operations
                cursor.execute("SELECT COUNT(*) FROM history")
                total_ops = cursor.fetchone()[0]

                # Most frequent cognitive state
                cursor.execute("""
                    SELECT cognitive_state, COUNT(*) as count
                    FROM history
                    GROUP BY cognitive_state
                    ORDER BY count DESC
                    LIMIT 1
                """)
                top_state_row = cursor.fetchone()
                top_state = top_state_row[0] if top_state_row else "None"

                return {
                    "total_operations": total_ops,
                    "dominant_state": top_state
                }
        except sqlite3.Error as e:
            logger.error(f"Failed to get session summary: {e}")
            return {}

    def search_history(
        self,
        query: Optional[str] = None,
        operation_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search history for operations matching query and/or type.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                sql = "SELECT * FROM history WHERE 1=1"
                params = []

                if operation_type:
                    sql += " AND operation = ?"
                    params.append(operation_type)

                if query:
                    # Simple LIKE search for now
                    sql += " AND (params LIKE ? OR result_summary LIKE ?)"
                    wildcard_query = f"%{query}%"
                    params.extend([wildcard_query, wildcard_query])

                sql += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)

                cursor.execute(sql, params)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Failed to search history: {e}")
            return []

    def get_focused_episodes(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve high-quality episodes (Focused state or low energy) for training.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT operation, params, result_summary
                    FROM history
                    WHERE cognitive_state = 'Focused' OR energy < 0.5
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve focused episodes: {e}")
            return []
