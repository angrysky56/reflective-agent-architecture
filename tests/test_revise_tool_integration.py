import asyncio
import json
import logging
import sys
from typing import Any, Dict

# Add project root to path
sys.path.append(".")

from src.server import call_tool, server_context

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_revise")

async def test_revise_tool():
    print("Initializing Server Context...")
    server_context.initialize()

    try:
        print("Testing 'revise' tool...")

        # Test case: Revise a belief to match evidence
        # Belief: "The cat is on the mat."
        # Evidence: "The cat is actually on the sofa."
        # Constraint: "Must mention the cat's location."

        arguments = {
            "belief": "The cat is on the mat.",
            "evidence": "The cat is actually on the sofa.",
            "constraints": ["The location must be accurate."]
        }

        # Call the tool
        # Note: call_tool is decorated, so we might need to call the original function or invoke it via the server wrapper.
        # The @server.call_tool() decorator registers it but might not change the signature if it's mcp.server.Server.
        # However, looking at server.py, it uses @server.call_tool().
        # We can try calling the function directly if the decorator preserves it,
        # or we might need to simulate the MCP call.
        # Let's try calling `call_tool` function defined in server.py directly.

        results = await call_tool("revise", arguments)

        print("\n=== Tool Output ===")
        for content in results:
            print(content.text)

            # Verify structure
            data = json.loads(content.text)
            if data["status"] == "success":
                print("\nSUCCESS: Tool executed successfully.")
                print(f"Strategy: {data['strategy']}")
                print(f"Revised Content: {data['revised_content']}")
            else:
                print(f"\nFAILURE: {data.get('message')}")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nCleaning up...")
        server_context.cleanup()

if __name__ == "__main__":
    asyncio.run(test_revise_tool())
