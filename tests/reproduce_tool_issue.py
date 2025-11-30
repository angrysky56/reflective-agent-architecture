
import asyncio
import logging
import os
import sys
from unittest.mock import MagicMock

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure logging
logging.basicConfig(level=logging.DEBUG)

async def main():
    print("Starting reproduction script...")

    try:
        from src.compass.adapters import RAALLMProvider
        from src.compass.compass_framework import COMPASS

        # Mock LLM Provider
        class MockLLM:
            def __init__(self):
                self.model_name = "mock-model"

            async def chat_completion(self, messages, stream=False, **kwargs):
                print(f"\n[MockLLM] Called with tools: {kwargs.get('tools')}")
                if kwargs.get('tools'):
                    print(f"[MockLLM] Tool count: {len(kwargs['tools'])}")

                # Return a valid JSON response that SLAP expects
                response = '{"action": "Test action", "confidence": 0.9, "reasoning": "Test reasoning"}'
                yield response

        # Mock MCP Client
        class MockMCP:
            def get_available_tools(self):
                print("\n[MockMCP] get_available_tools called")
                return [
                    MagicMock(
                        name="test_tool",
                        description="A test tool",
                        inputSchema={"type": "object", "properties": {"query": {"type": "string"}}}
                    )
                ]

            async def call_tool(self, name, arguments):
                return "Tool result"

        # Initialize COMPASS with mocks
        print("\nInitializing COMPASS...")
        compass = COMPASS(llm_provider=MockLLM(), mcp_client=MockMCP())

        # Force enable tools
        compass.integrated_intelligence.config.enable_tools = True
        print(f"Tools enabled: {compass.integrated_intelligence.config.enable_tools}")

        # Run process_task
        print("\nRunning process_task...")
        # We need to mock the other components that COMPASS uses internally if they are not mocked by default
        # But COMPASS __init__ creates them. Let's hope they don't crash without real dependencies.
        # We might need to mock shape_processor, etc.

        # Mock SHAPE processor to avoid real LLM calls
        async def mock_shape(text):
            return {
                "original": text,
                "intent": "Test intent",
                "entities": [],
                "constraints": [],
                "implicit_goals": []
            }
        compass.shape_processor.process_user_input = mock_shape

        # Mock SLAP to avoid real LLM calls and return a valid plan
        async def mock_slap(task, context, representation_type=None):
            return {
                "steps": ["Step 1"],
                "reasoning": "Plan reasoning"
            }
        compass.slap_pipeline.create_reasoning_plan = mock_slap
        
        # Verify /tmp writability
        try:
            with open("/tmp/test_write.txt", "w") as f:
                f.write("test")
            print("✅ /tmp is writable")
        except Exception as e:
            print(f"❌ /tmp is NOT writable: {e}")

        # Run it
        result = await compass.process_task("Test task")
        print("\nResult:", result)

        # Check for log files
        print("\nChecking log files:")
        for log_file in ["/tmp/process_task_CALLED.txt", "/tmp/make_decision_CALLED.txt", "/tmp/integrated_intelligence_debug.log"]:
            if os.path.exists(log_file):
                print(f"✅ {log_file} exists")
                # Print content
                with open(log_file, 'r') as f:
                    print(f"--- Content of {log_file} ---")
                    print(f.read())
                    print("-----------------------------")
            else:
                print(f"❌ {log_file} does NOT exist")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
