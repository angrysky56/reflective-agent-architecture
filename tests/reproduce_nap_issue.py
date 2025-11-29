import logging
from unittest.mock import MagicMock, patch

from src.integration.sleep_cycle import SleepCycle

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_nap_empty_history():
    print("\n--- Testing Nap with Empty History ---")
    # Mock workspace
    mock_workspace = MagicMock()

    # Initialize SleepCycle with mock workspace
    sleep = SleepCycle(workspace=mock_workspace)

    # Mock WorkHistory to return empty list
    sleep.history.get_focused_episodes = MagicMock(return_value=[])

    # Run dream
    result = sleep.dream(epochs=5)
    print(f"Result: {result}")

def test_nap_with_history():
    print("\n--- Testing Nap with History ---")
    # Mock workspace
    mock_workspace = MagicMock()

    # Initialize SleepCycle
    sleep = SleepCycle(workspace=mock_workspace)

    # Mock WorkHistory to return some episodes
    episodes = [
        {"operation": "op1", "params": "p1", "result_summary": "r1"},
        {"operation": "op2", "params": "p2", "result_summary": "r2"}
    ]
    sleep.history.get_focused_episodes = MagicMock(return_value=episodes)

    # Mock tokenizer and processor to avoid actual training overhead
    sleep.tokenizer = MagicMock()
    # Mock input_ids as a mock that returns itself on .to() and .clone()
    mock_tensor = MagicMock()
    mock_tensor.to.return_value = mock_tensor
    mock_tensor.clone.return_value = mock_tensor

    sleep.tokenizer.return_value = MagicMock(input_ids=mock_tensor)
    sleep.processor = MagicMock()
    sleep.processor.train_step.return_value = 0.5

    # Run dream
    result = sleep.dream(epochs=2)
    print(f"Result: {result}")

if __name__ == "__main__":
    test_nap_empty_history()
    test_nap_with_history()
