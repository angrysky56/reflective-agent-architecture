
import os
import sys
from unittest.mock import MagicMock

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.director.director_core import DirectorConfig, DirectorMVP


def test_adaptive_temperature():
    print("Testing Adaptive Temperature Logic...")

    # Mock dependencies
    manifold = MagicMock()
    mcp_client = MagicMock()
    continuity_service = MagicMock()
    llm_provider = MagicMock()

    config = DirectorConfig(
        adaptive_temp_min=0.2,
        adaptive_temp_max=0.8
    )

    director = DirectorMVP(
        manifold=manifold,
        config=config,
        embedding_fn=lambda x: [0.1]*10,
        mcp_client=mcp_client,
        continuity_service=continuity_service,
        llm_provider=llm_provider
    )

    # Test Case 1: Stable State (Low Energy)
    # Energy = -5.0 (Very Stable) -> Should be min_temp (0.2)
    director.latest_cognitive_state = ("Stable", -5.0)
    temp = director.get_adaptive_temperature()
    print(f"Energy: -5.0 -> Temp: {temp:.2f} (Expected: 0.20)")
    assert abs(temp - 0.2) < 0.01

    # Test Case 2: Unstable State (High Energy)
    # Energy = 0.0 (Very Unstable) -> Should be max_temp (0.8)
    director.latest_cognitive_state = ("Unstable", 0.0)
    temp = director.get_adaptive_temperature()
    print(f"Energy: 0.0 -> Temp: {temp:.2f} (Expected: 0.80)")
    assert abs(temp - 0.8) < 0.01

    # Test Case 3: Neutral State (Mid Energy)
    # Energy = -2.5 (Midpoint) -> Should be mid_temp (0.5)
    director.latest_cognitive_state = ("Neutral", -2.5)
    temp = director.get_adaptive_temperature()
    print(f"Energy: -2.5 -> Temp: {temp:.2f} (Expected: 0.50)")
    assert abs(temp - 0.5) < 0.01

    # Test Case 4: Out of bounds (Very Low Energy)
    # Energy = -10.0 -> Should be clamped to min_temp (0.2)
    director.latest_cognitive_state = ("SuperStable", -10.0)
    temp = director.get_adaptive_temperature()
    print(f"Energy: -10.0 -> Temp: {temp:.2f} (Expected: 0.20)")
    assert abs(temp - 0.2) < 0.01

    # Test Case 5: Out of bounds (Positive Energy)
    # Energy = 1.0 -> Should be clamped to max_temp (0.8)
    director.latest_cognitive_state = ("SuperUnstable", 1.0)
    temp = director.get_adaptive_temperature()
    print(f"Energy: 1.0 -> Temp: {temp:.2f} (Expected: 0.80)")
    assert abs(temp - 0.8) < 0.01

    print("\nAll tests passed!")

if __name__ == "__main__":
    test_adaptive_temperature()
