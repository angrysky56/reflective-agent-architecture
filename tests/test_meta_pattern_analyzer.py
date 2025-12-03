"""
Tests for MetaPatternAnalyzer.
"""

import time
from typing import List

import pytest

from src.director.intervention_tracker import InterventionOutcome, InterventionRecord
from src.director.meta_pattern_analyzer import MetaPatternAnalyzer


def create_mock_record(
    threshold: float,
    quality: float,
    state: str = "Neutral",
    ep_id: str = "1"
) -> InterventionRecord:
    """Helper to create a mock record with outcome."""
    record = InterventionRecord(
        episode_id=ep_id,
        timestamp=time.time(),
        entropy_before=threshold + 0.1, # Triggered because > threshold
        energy_before=10.0,
        cognitive_state_before=state,
        goal_before="Goal",
        intervention_type="search",
        intervention_source="entropy",
        entropy_threshold_used=threshold
    )
    record.outcome = InterventionOutcome(
        timestamp=time.time(),
        entropy_after=threshold - 0.5,
        energy_after=5.0,
        task_success=quality > 0.5,
        outcome_quality=quality,
        entropy_delta=-0.5
    )
    return record


def test_insufficient_samples():
    analyzer = MetaPatternAnalyzer(min_samples=10)
    records = [create_mock_record(2.0, 0.8) for _ in range(5)]
    insights = analyzer.analyze(records)
    assert len(insights) == 0


def test_threshold_optimization_raise():
    """
    Scenario: Low threshold interventions (e.g. 1.5) have LOW quality (0.2).
    High threshold interventions (e.g. 2.5) have HIGH quality (0.9).
    Analyzer should suggest RAISING threshold.
    """
    analyzer = MetaPatternAnalyzer(min_samples=10)
    records = []

    # 10 records with low threshold (1.5) and poor quality (0.2)
    for i in range(10):
        records.append(create_mock_record(1.5, 0.2, ep_id=f"low_{i}"))

    # 10 records with high threshold (2.5) and good quality (0.9)
    for i in range(10):
        records.append(create_mock_record(2.5, 0.9, ep_id=f"high_{i}"))

    insights = analyzer.analyze(records)

    # Should find a threshold optimization pattern
    threshold_insights = [i for i in insights if i.pattern_type == "threshold_optimization"]
    assert len(threshold_insights) > 0

    insight = threshold_insights[0]
    assert "Raise threshold" in insight.recommendation
    assert insight.suggested_adjustment["threshold_multiplier"] > 1.0


def test_threshold_optimization_lower():
    """
    Scenario: Low threshold interventions have HIGH quality.
    High threshold interventions have LOW quality (maybe too late?).
    Analyzer should suggest LOWERING threshold (or at least favoring low).
    """
    analyzer = MetaPatternAnalyzer(min_samples=10)
    records = []

    # 10 records with low threshold (1.5) and good quality (0.9)
    for i in range(10):
        records.append(create_mock_record(1.5, 0.9, ep_id=f"low_{i}"))

    # 10 records with high threshold (2.5) and poor quality (0.2)
    for i in range(10):
        records.append(create_mock_record(2.5, 0.2, ep_id=f"high_{i}"))

    insights = analyzer.analyze(records)

    threshold_insights = [i for i in insights if i.pattern_type == "threshold_optimization"]
    assert len(threshold_insights) > 0

    insight = threshold_insights[0]
    assert "Lower threshold" in insight.recommendation
    assert insight.suggested_adjustment["threshold_multiplier"] < 1.0


def test_state_specific_looping():
    """
    Scenario: 'Looping' state has consistently poor outcomes compared to global average.
    Analyzer should suggest increasing sensitivity (lowering threshold) for this state.
    """
    analyzer = MetaPatternAnalyzer(min_samples=10)
    records = []

    # 15 records: 10 Normal (good), 5 Looping (bad)
    for i in range(10):
        records.append(create_mock_record(2.0, 0.9, state="Normal", ep_id=f"norm_{i}"))

    for i in range(10): # Need enough samples
        records.append(create_mock_record(2.0, 0.2, state="Looping", ep_id=f"loop_{i}"))

    insights = analyzer.analyze(records)

    state_insights = [i for i in insights if i.pattern_type == "state_specific_Looping"]
    assert len(state_insights) > 0

    insight = state_insights[0]
    assert "Looping" in insight.recommendation
    assert insight.suggested_adjustment["threshold_multiplier"] < 1.0
