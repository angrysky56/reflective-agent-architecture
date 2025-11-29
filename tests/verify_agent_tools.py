import asyncio
import unittest
from typing import Any, AsyncGenerator, Dict, List
from unittest.mock import AsyncMock, MagicMock


# Mock RAALLMProvider
class MockLLMProvider:
    def __init__(self):
        self.calls = []

    async def chat_completion(self, messages, stream=False, tools=None):
        self.calls.append({"messages": messages, "tools": tools})

        # Simulate a tool call in the first turn
        if len(self.calls) == 1:
            yield '{"tool_calls": [{"function": {"name": "consult_compass", "arguments": "{\\"task\\": \\"test task\\"}"}}]}'
        else:
            # Simulate final response
            yield "I have consulted compass."

# Mock Tool Executor
async def mock_tool_executor(name: str, args: Dict[str, Any]):
    if name == "consult_compass":
        return "COMPASS Analysis Result"
    return f"Unknown tool {name}"

class TestAgentTools(unittest.IsolatedAsyncioTestCase):
    async def test_agent_tool_execution(self):
        from src.integration.agent_factory import AgentFactory

        provider = MockLLMProvider()
        factory = AgentFactory(llm_provider=provider, tool_executor=mock_tool_executor)

        # Spawn an agent
        agent_name = factory.spawn_agent("Tension Loop", "Conflict context")

        # Execute agent
        response = await factory.execute_agent(agent_name, {"query": "Resolve this"})

        # Verify tool was called
        # We check if the LLM provider received the tool result in the second call
        self.assertEqual(len(provider.calls), 2)
        last_messages = provider.calls[1]["messages"]

        # Check for tool result message
        tool_result_msg = next((m for m in last_messages if m.role == "user" and "Tool 'consult_compass' result" in m.content), None)
        self.assertIsNotNone(tool_result_msg)
        self.assertIn("COMPASS Analysis Result", tool_result_msg.content)

        print(f"Agent Response: {response}")

if __name__ == "__main__":
    unittest.main()
