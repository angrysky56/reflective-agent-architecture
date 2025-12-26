"""
Intervention Tracker

Implements the memory system for Recursive Observer dynamics.
Tracks Layer 4 interventions and their outcomes to enable reflexive closure.
"""

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class InterventionOutcome:
    """Result of an intervention."""

    timestamp: float
    entropy_after: float
    energy_after: float
    task_success: bool
    outcome_quality: float  # 0.0 to 1.0 continuous score
    entropy_delta: float
    convergence_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InterventionRecord:
    """Complete record of a single intervention episode."""

    episode_id: str
    timestamp: float

    # Layer 3 State (Before)
    entropy_before: float
    energy_before: float
    cognitive_state_before: str
    goal_before: str

    # Layer 4 Intervention
    intervention_type: str  # e.g., "search", "reframe"
    intervention_source: str  # e.g., "entropy", "sheaf", "compass"
    entropy_threshold_used: float
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Outcome (populated later)
    outcome: Optional[InterventionOutcome] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Handle datetime serialization if needed, but we use float timestamps here
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InterventionRecord":
        """Create from dictionary."""
        if data.get("outcome"):
            data["outcome"] = InterventionOutcome(**data["outcome"])
        return cls(**data)


class InterventionTracker:
    """
    Tracks interventions and their outcomes.

    Thread-safe storage for intervention history.
    Persists to disk to enable long-term learning.
    """

    def __init__(
        self,
        persistence_path: Optional[Path] = None,
        max_memory: int = 1000,
        autosave_interval: int = 10,
    ):
        """
        Initialize tracker.

        Args:
            persistence_path: Path to JSON file for saving history
            max_memory: Maximum number of records to keep in memory
            autosave_interval: Save to disk after this many updates
        """
        self.persistence_path = (
            persistence_path or Path.home() / ".raa" / "intervention_history.json"
        )
        self.max_memory = max_memory
        self.autosave_interval = autosave_interval

        self._history: Dict[str, InterventionRecord] = {}
        self._lock = threading.Lock()
        self._unsaved_changes = 0

        # Ensure directory exists
        if self.persistence_path:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_history()

    def start_intervention(
        self,
        episode_id: str,
        entropy: float,
        energy: float,
        cognitive_state: str,
        goal: str,
        intervention_type: str,
        intervention_source: str,
        threshold: float,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> InterventionRecord:
        """
        Record the start of an intervention.

        Args:
            episode_id: Unique ID for this episode
            entropy: Current entropy value
            energy: Current energy value
            cognitive_state: Current cognitive state label
            goal: Current goal description
            intervention_type: Type of intervention (search, reframe, etc.)
            intervention_source: What triggered it (entropy, sheaf, etc.)
            threshold: The threshold that was exceeded
            parameters: Additional parameters used

        Returns:
            The created record
        """
        record = InterventionRecord(
            episode_id=episode_id,
            timestamp=time.time(),
            entropy_before=entropy,
            energy_before=energy,
            cognitive_state_before=cognitive_state,
            goal_before=goal,
            intervention_type=intervention_type,
            intervention_source=intervention_source,
            entropy_threshold_used=threshold,
            parameters=parameters or {},
        )

        with self._lock:
            self._history[episode_id] = record
            self._prune_history()
            self._mark_change()

        return record

    def finish_intervention(
        self,
        episode_id: str,
        entropy_after: float,
        energy_after: float,
        task_success: bool,
        outcome_quality: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[InterventionRecord]:
        """
        Record the outcome of an intervention.

        Args:
            episode_id: ID of the episode to update
            entropy_after: Entropy after intervention
            energy_after: Energy after intervention
            task_success: Whether the immediate task succeeded
            outcome_quality: Continuous quality score (0.0-1.0)
            metadata: Additional outcome metadata

        Returns:
            Updated record or None if not found
        """
        with self._lock:
            if episode_id not in self._history:
                logger.warning(f"Attempted to finish unknown intervention: {episode_id}")
                return None

            record = self._history[episode_id]

            # Calculate deltas
            entropy_delta = entropy_after - record.entropy_before
            convergence_time = time.time() - record.timestamp

            outcome = InterventionOutcome(
                timestamp=time.time(),
                entropy_after=entropy_after,
                energy_after=energy_after,
                task_success=task_success,
                outcome_quality=outcome_quality,
                entropy_delta=entropy_delta,
                convergence_time=convergence_time,
                metadata=metadata or {},
            )

            record.outcome = outcome
            self._mark_change()

            return record

    def get_record(self, episode_id: str) -> Optional[InterventionRecord]:
        """Get a specific record."""
        with self._lock:
            return self._history.get(episode_id)

    def get_recent_interventions(
        self, limit: int = 100, min_quality: Optional[float] = None
    ) -> List[InterventionRecord]:
        """
        Get recent interventions, optionally filtered.

        Args:
            limit: Max records to return
            min_quality: Filter for outcomes with at least this quality

        Returns:
            List of records sorted by timestamp (newest first)
        """
        with self._lock:
            records = list(self._history.values())

        # Sort by timestamp descending
        records.sort(key=lambda r: r.timestamp, reverse=True)

        # Filter
        if min_quality is not None:
            records = [r for r in records if r.outcome and r.outcome.outcome_quality >= min_quality]

        return records[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        with self._lock:
            total = len(self._history)
            completed = sum(1 for r in self._history.values() if r.outcome)

            if completed == 0:
                return {"total": total, "completed": 0, "success_rate": 0.0}

            successes = sum(
                1 for r in self._history.values() if r.outcome and r.outcome.task_success
            )
            avg_quality = (
                sum(r.outcome.outcome_quality for r in self._history.values() if r.outcome)
                / completed
            )

            return {
                "total": total,
                "completed": completed,
                "success_rate": successes / completed,
                "avg_quality": avg_quality,
            }

    def _prune_history(self) -> None:
        """Remove old records if exceeding max_memory."""
        if len(self._history) > self.max_memory:
            # Sort by timestamp
            sorted_ids = sorted(self._history.keys(), key=lambda k: self._history[k].timestamp)

            # Remove oldest
            to_remove = len(self._history) - self.max_memory
            for i in range(to_remove):
                del self._history[sorted_ids[i]]

    def _mark_change(self) -> None:
        """Mark a change and potentially save."""
        self._unsaved_changes += 1
        if self._unsaved_changes >= self.autosave_interval:
            self._save_history()

    def _save_history(self) -> None:
        """Save history to disk."""
        if not self.persistence_path:
            return

        try:
            data = {
                "version": "1.0",
                "timestamp": time.time(),
                "records": [r.to_dict() for r in self._history.values()],
            }

            with open(self.persistence_path, "w") as f:
                json.dump(data, f, indent=2)

            self._unsaved_changes = 0
            logger.debug(f"Saved {len(self._history)} intervention records")

        except Exception as e:
            logger.error(f"Failed to save intervention history: {e}")

    def _load_history(self) -> None:
        """Load history from disk."""
        if not self.persistence_path or not self.persistence_path.exists():
            return

        try:
            with open(self.persistence_path, "r") as f:
                data = json.load(f)

            for record_data in data.get("records", []):
                try:
                    record = InterventionRecord.from_dict(record_data)
                    self._history[record.episode_id] = record
                except Exception as e:
                    logger.warning(f"Failed to load record: {e}")

            logger.info(f"Loaded {len(self._history)} intervention records")

        except Exception as e:
            logger.error(f"Failed to load intervention history: {e}")
