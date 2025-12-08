from unittest.mock import MagicMock

import pytest
import torch

from src.integration.sleep_cycle import SleepCycle


def test_sleep_cycle():
    """Verify Sleep Cycle functionality."""
    # Mock workspace
    workspace = MagicMock()

    # Initialize Sleep Cycle
    sleep = SleepCycle(db_path=":memory:", workspace=workspace)

    # Mock tokenizer to avoid download
    sleep.tokenizer = MagicMock()
    sleep.tokenizer.return_value.input_ids = torch.randint(0, 100, (1, 16))
    sleep.tokenizer.pad_token = "pad"

    # Mock DB interaction
    sleep.history.get_focused_episodes = MagicMock(return_value=[
        {"operation": "op", "params": "params", "result_summary": "result"}
    ])

    # Run Dream
    results = sleep.dream(epochs=1)

    # Verify Replay
    epoch_result = results["sleep_cycle_results"][0]
    assert epoch_result["replay"]["steps"] > 0
    assert epoch_result["replay"]["avg_loss"] > 0

    # Verify Crystallization
    assert "new_tools_created" in epoch_result["crystallization"]

if __name__ == "__main__":
    test_sleep_cycle()
