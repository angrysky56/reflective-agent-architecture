"""
Tests for ReflexiveClosureEngine.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from src.director.adaptive_criterion import AdaptiveCriterion
from src.director.intervention_tracker import InterventionOutcome, InterventionRecord, InterventionTracker
from src.director.meta_pattern_analyzer import MetaPatternAnalyzer, PatternInsight
from src.director.reflexive_closure_engine import ReflexiveClosureEngine


@pytest.fixture
def mock_components():
    tracker = MagicMock(spec=InterventionTracker)
    analyzer = MagicMock(spec=MetaPatternAnalyzer)
    criterion = MagicMock(spec=AdaptiveCriterion)

    # Setup default behaviors
    tracker.start_intervention.return_value = InterventionRecord(
        episode_id="test_ep", timestamp=0, entropy_before=0, energy_before=0,
        cognitive_state_before="", goal_before="", intervention_type="", intervention_source="", entropy_threshold_used=0
    )
    criterion.get_threshold.return_value = 2.0

    return tracker, analyzer, criterion


def test_initialization(mock_components):
    tracker, analyzer, criterion = mock_components
    engine = ReflexiveClosureEngine(tracker, analyzer, criterion)
    assert engine.tracker == tracker
    assert engine.analyzer == analyzer
    assert engine.criterion == criterion


def test_get_threshold_no_exploration(mock_components):
    tracker, analyzer, criterion = mock_components
    engine = ReflexiveClosureEngine(tracker, analyzer, criterion, exploration_rate=0.0)

    threshold = engine.get_threshold("Neutral")
    assert threshold == 2.0
    criterion.get_threshold.assert_called_with("Neutral")


def test_get_threshold_with_exploration(mock_components):
    tracker, analyzer, criterion = mock_components
    engine = ReflexiveClosureEngine(tracker, analyzer, criterion, exploration_rate=1.0) # Always explore

    with patch('random.uniform', return_value=1.1):
        threshold = engine.get_threshold("Neutral")
        assert threshold == 2.2  # 2.0 * 1.1


def test_analysis_triggering(mock_components):
    tracker, analyzer, criterion = mock_components
    # Set interval to 2
    engine = ReflexiveClosureEngine(tracker, analyzer, criterion, analysis_interval=2)

    # Mock analyzer to return an insight
    insight = PatternInsight("test", 0.9, "rec", {}, 1)
    analyzer.analyze.return_value = [insight]
    tracker.get_recent_interventions.return_value = []

    # 1. First intervention
    engine.record_intervention_end(episode_id="1", entropy_after=1.0, energy_after=1.0, task_success=True, outcome_quality=1.0)
    analyzer.analyze.assert_not_called()

    # 2. Second intervention - should trigger
    engine.record_intervention_end(episode_id="2", entropy_after=1.0, energy_after=1.0, task_success=True, outcome_quality=1.0)
    analyzer.analyze.assert_called_once()
    criterion.update.assert_called_with(insight)

    # 3. Reset check
    analyzer.analyze.reset_mock()
    engine.record_intervention_end(episode_id="3", entropy_after=1.0, energy_after=1.0, task_success=True, outcome_quality=1.0)
    analyzer.analyze.assert_not_called()
