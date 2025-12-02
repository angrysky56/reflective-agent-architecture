import logging
from unittest.mock import MagicMock, patch

import pytest
from openai import RateLimitError
from tenacity import RetryError

# Configure logging to see retry logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.llm.openrouter_provider import OpenRouterProvider


def test_openrouter_retry_logic():
    """
    Test that OpenRouterProvider retries on RateLimitError.
    """
    provider = OpenRouterProvider(model_name="test-model", api_key="test-key")

    # Mock the OpenAI client
    with patch("src.llm.openrouter_provider.OpenAI") as MockOpenAI:
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client

        # Configure the mock to raise RateLimitError
        # We need to simulate a response object for the exception if needed,
        # but RateLimitError usually takes message, response, body.
        # Let's just mock the exception raising.

        # Create a mock response object required by RateLimitError
        mock_response = MagicMock()
        mock_response.headers = {"Retry-After": "1"}

        error = RateLimitError(message="Rate limit exceeded", response=mock_response, body=None)

        mock_client.chat.completions.create.side_effect = error

        # Call generate - it should retry 5 times then fail
        # We expect a RetryError from tenacity wrapping the RateLimitError
        with pytest.raises(RetryError):
            provider.generate("system", "user")

        # Verify call count
        # Tenacity stop_after_attempt(5) means it calls 5 times total
        assert mock_client.chat.completions.create.call_count == 5
        print(f"\nVerified: Called {mock_client.chat.completions.create.call_count} times before failing.")

if __name__ == "__main__":
    # Manually run if executed directly
    try:
        test_openrouter_retry_logic()
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
