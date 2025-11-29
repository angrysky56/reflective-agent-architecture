import asyncio
import json
import unittest
from unittest.mock import MagicMock, patch


# Mock Ollama ToolCall object
class MockToolCall:
    def __init__(self, name, args):
        self.function = {"name": name, "arguments": args}

    def model_dump(self):
        return {"function": self.function}

class TestAdapterSerialization(unittest.IsolatedAsyncioTestCase):
    async def test_serialization(self):
        from src.compass.adapters import Message, RAALLMProvider

        provider = RAALLMProvider()

        # Mock ollama.chat to return a generator with a tool call object
        with patch("ollama.chat") as mock_chat:
            # Create a mock response chunk with a tool call object
            tool_call = MockToolCall("test_tool", "{}")
            mock_chunk = {"message": {"role": "assistant", "tool_calls": [tool_call]}}

            # Mock return value
            mock_chat.return_value = [mock_chunk]

            messages = [Message(role="user", content="test")]

            # Run
            results = []
            async for chunk in provider.chat_completion(messages):
                results.append(chunk)

            # Verify
            self.assertTrue(len(results) > 0)
            last_chunk = results[-1]
            print(f"Last chunk: {last_chunk}")

            # Should be valid JSON
            data = json.loads(last_chunk)
            self.assertIn("tool_calls", data)
            self.assertEqual(data["tool_calls"][0]["function"]["name"], "test_tool")

if __name__ == "__main__":
    unittest.main()
