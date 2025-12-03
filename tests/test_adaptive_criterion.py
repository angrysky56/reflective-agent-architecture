"""
Tests for AdaptiveCriterion.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from src.director.adaptive_criterion import AdaptiveCriterion
from src.director.meta_pattern_analyzer import PatternInsight


@pytest.fixture
def temp_criterion():
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "criterion.json"
        criterion = AdaptiveCriterion(
            initial_threshold=2.0,
            min_threshold=1.0,
            max_threshold=3.0,
            persistence_path=path
        )
        yield criterion, path


def test_initialization(temp_criterion):
    criterion, path = temp_criterion
    assert criterion.state.base_threshold == 2.0
    assert criterion.get_threshold() == 2.0


def test_persistence(temp_criterion):
    criterion, path = temp_criterion

    # Modify state
    criterion.state.base_threshold = 2.5
    criterion._save()

    # Reload
    new_criterion = AdaptiveCriterion(persistence_path=path)
    assert new_criterion.state.base_threshold == 2.5


def test_update_threshold(temp_criterion):
    criterion, _ = temp_criterion

    insight = PatternInsight(
        pattern_type="threshold_optimization",
        confidence=0.8,
        recommendation="Raise threshold",
        suggested_adjustment={"threshold_multiplier": 1.1},
        evidence_count=10
    )

    updated = criterion.update(insight)

    assert updated is True
    assert criterion.state.base_threshold == 2.2  # 2.0 * 1.1


def test_update_bounds_check(temp_criterion):
    criterion, _ = temp_criterion
    # Max is 3.0

    insight = PatternInsight(
        pattern_type="threshold_optimization",
        confidence=0.8,
        recommendation="Raise threshold huge",
        suggested_adjustment={"threshold_multiplier": 2.0}, # Would be 4.0
        evidence_count=10
    )

    criterion.update(insight)
    assert criterion.state.base_threshold == 3.0  # Clamped to max


def test_state_specific_override(temp_criterion):
    criterion, _ = temp_criterion

    # 1. Verify default
    assert criterion.get_threshold("Looping") == 2.0

    # 2. Apply override
    insight = PatternInsight(
        pattern_type="state_specific_Looping",
        confidence=0.8,
        recommendation="Lower for Looping",
        suggested_adjustment={"state": "Looping", "threshold_multiplier": 0.5},
        evidence_count=10
    )

    criterion.update(insight)

    # 3. Verify override applied
    assert criterion.get_threshold("Looping") == 1.0 # 2.0 * 0.5
    assert criterion.get_threshold("Neutral") == 2.0 # Unchanged

    # 4. Verify partial match
    assert criterion.get_threshold("Looping (Stuck)") == 1.0
