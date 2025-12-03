import pytest

from src.substrate.entropy import CognitiveState, EntropyMonitor


def test_entropy_calculation():
    monitor = EntropyMonitor()

    # Case 1: Uniform distribution (High Entropy)
    # 4 items, p=0.25 each. H = -4 * (0.25 * log2(0.25)) = -1 * -2 = 2.0
    counts = {"A": 10, "B": 10, "C": 10, "D": 10}
    entropy = monitor.update_from_counts(counts)
    assert entropy == 2.0

    # Case 2: Peaked distribution (Low Entropy)
    # A is dominant.
    counts = {"A": 100, "B": 1, "C": 1, "D": 1}
    entropy = monitor.update_from_counts(counts)
    assert entropy < 1.0 # Should be low

    # Case 3: Single item (Zero Entropy)
    counts = {"A": 100}
    entropy = monitor.update_from_counts(counts)
    assert entropy == 0.0

def test_state_transitions():
    # Thresholds: Focus > 2.5, Explore < 1.0
    monitor = EntropyMonitor(focus_threshold=2.5, explore_threshold=1.0)

    # Initial state
    assert monitor.state == CognitiveState.EXPLORE

    # 1. High Entropy -> Trigger FOCUS
    # 8 items uniform -> H=3.0
    counts = {str(i): 10 for i in range(8)}
    monitor.update_from_counts(counts)
    assert monitor.current_entropy == 3.0
    assert monitor.state == CognitiveState.FOCUS

    # 2. Medium Entropy -> Maintain FOCUS (Hysteresis/Stability)
    # 4 items uniform -> H=2.0 (Between 1.0 and 2.5)
    counts = {str(i): 10 for i in range(4)}
    monitor.update_from_counts(counts)
    assert monitor.current_entropy == 2.0
    assert monitor.state == CognitiveState.FOCUS # Should stay in FOCUS

    # 3. Low Entropy -> Trigger EXPLORE
    # 1 item -> H=0.0
    counts = {"A": 100}
    monitor.update_from_counts(counts)
    assert monitor.current_entropy == 0.0
    assert monitor.state == CognitiveState.EXPLORE

def test_empty_counts():
    monitor = EntropyMonitor()
    entropy = monitor.update_from_counts({})
    assert entropy == 0.0
