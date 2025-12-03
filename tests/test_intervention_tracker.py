"""
Tests for InterventionTracker.
"""

import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from src.director.intervention_tracker import InterventionTracker


@pytest.fixture
def temp_tracker():
    """Create a tracker with a temporary persistence path."""
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "history.json"
        tracker = InterventionTracker(persistence_path=path, autosave_interval=1)
        yield tracker, path


def test_initialization(temp_tracker):
    tracker, path = temp_tracker
    assert tracker.persistence_path == path
    assert len(tracker._history) == 0


def test_start_intervention(temp_tracker):
    tracker, _ = temp_tracker

    record = tracker.start_intervention(
        episode_id="ep1",
        entropy=2.5,
        energy=10.0,
        cognitive_state="Looping",
        goal="Test Goal",
        intervention_type="search",
        intervention_source="entropy",
        threshold=2.0,
        parameters={"k": 5}
    )

    assert record.episode_id == "ep1"
    assert record.entropy_before == 2.5
    assert record.intervention_type == "search"
    assert record.parameters == {"k": 5}

    # Check it's in history
    stored = tracker.get_record("ep1")
    assert stored == record


def test_finish_intervention(temp_tracker):
    tracker, _ = temp_tracker

    # Start
    tracker.start_intervention(
        episode_id="ep1",
        entropy=2.5,
        energy=10.0,
        cognitive_state="Looping",
        goal="Test Goal",
        intervention_type="search",
        intervention_source="entropy",
        threshold=2.0
    )

    # Finish
    updated = tracker.finish_intervention(
        episode_id="ep1",
        entropy_after=1.5,
        energy_after=5.0,
        task_success=True,
        outcome_quality=0.9,
        metadata={"notes": "good"}
    )

    assert updated is not None
    assert updated.outcome is not None
    assert updated.outcome.entropy_after == 1.5
    assert updated.outcome.task_success is True
    assert updated.outcome.entropy_delta == -1.0  # 1.5 - 2.5
    assert updated.outcome.metadata == {"notes": "good"}


def test_finish_unknown_intervention(temp_tracker):
    tracker, _ = temp_tracker
    result = tracker.finish_intervention(
        episode_id="unknown",
        entropy_after=1.0,
        energy_after=1.0,
        task_success=True,
        outcome_quality=1.0
    )
    assert result is None


def test_get_recent_interventions(temp_tracker):
    tracker, _ = temp_tracker

    # Create 3 records with different timestamps
    for i in range(3):
        tracker.start_intervention(
            episode_id=f"ep{i}",
            entropy=2.0,
            energy=10.0,
            cognitive_state="State",
            goal="Goal",
            intervention_type="search",
            intervention_source="entropy",
            threshold=2.0
        )
        time.sleep(0.01)  # Ensure timestamp diff

        if i == 1:  # Mark middle one as high quality
            tracker.finish_intervention(
                episode_id=f"ep{i}",
                entropy_after=1.0,
                energy_after=5.0,
                task_success=True,
                outcome_quality=0.9
            )
        elif i == 2: # Mark last one as low quality
             tracker.finish_intervention(
                episode_id=f"ep{i}",
                entropy_after=1.0,
                energy_after=5.0,
                task_success=False,
                outcome_quality=0.2
            )

    # Test sorting (newest first)
    recent = tracker.get_recent_interventions(limit=10)
    assert len(recent) == 3
    assert recent[0].episode_id == "ep2"
    assert recent[2].episode_id == "ep0"

    # Test filtering
    high_quality = tracker.get_recent_interventions(min_quality=0.8)
    assert len(high_quality) == 1
    assert high_quality[0].episode_id == "ep1"


def test_persistence(temp_tracker):
    tracker, path = temp_tracker

    # Create and save
    tracker.start_intervention(
        episode_id="ep1",
        entropy=2.5,
        energy=10.0,
        cognitive_state="Looping",
        goal="Test Goal",
        intervention_type="search",
        intervention_source="entropy",
        threshold=2.0
    )
    # Force save (autosave is 1, so it should have saved)

    assert path.exists()

    # Load new tracker from same path
    new_tracker = InterventionTracker(persistence_path=path)
    loaded_record = new_tracker.get_record("ep1")

    assert loaded_record is not None
    assert loaded_record.episode_id == "ep1"
    assert loaded_record.entropy_before == 2.5


def test_thread_safety(temp_tracker):
    tracker, _ = temp_tracker

    def worker(worker_id):
        for i in range(100):
            ep_id = f"w{worker_id}_{i}"
            tracker.start_intervention(
                episode_id=ep_id,
                entropy=2.0,
                energy=10.0,
                cognitive_state="State",
                goal="Goal",
                intervention_type="search",
                intervention_source="entropy",
                threshold=2.0
            )
            tracker.finish_intervention(
                episode_id=ep_id,
                entropy_after=1.0,
                energy_after=5.0,
                task_success=True,
                outcome_quality=0.8
            )

    threads = []
    for i in range(5):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    stats = tracker.get_stats()
    assert stats["total"] == 500
    assert stats["completed"] == 500


def test_pruning(temp_tracker):
    # Create tracker with small memory
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "history.json"
        tracker = InterventionTracker(
            persistence_path=path,
            max_memory=5,
            autosave_interval=100
        )

        for i in range(10):
            tracker.start_intervention(
                episode_id=f"ep{i}",
                entropy=2.0,
                energy=10.0,
                cognitive_state="State",
                goal="Goal",
                intervention_type="search",
                intervention_source="entropy",
                threshold=2.0
            )
            time.sleep(0.001)

        assert len(tracker._history) == 5

        # Should have kept the newest ones (5-9)
        assert tracker.get_record("ep9") is not None
        assert tracker.get_record("ep0") is None
