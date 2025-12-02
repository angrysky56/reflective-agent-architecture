
import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.server import call_tool, get_raa_context


async def main():
    print("Initializing RAA Context...")
    ctx = get_raa_context()

    print("Calling consult_compass...")
    try:
        result = await call_tool("consult_compass", {"task": "What is 2+2?"})
        print("\nResult:")
        for content in result:
            print(content.text)

        if "[SUCCESS]" in result[0].text:
            print("\nVERIFICATION PASSED: COMPASS returned success.")
        else:
            print("\nVERIFICATION FAILED: COMPASS did not return success.")

    except Exception as e:
        print(f"\nVERIFICATION FAILED: Exception occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
