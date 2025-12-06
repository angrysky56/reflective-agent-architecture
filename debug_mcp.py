
import asyncio
import os
import sys

from src.dashboard.mcp_client_wrapper import get_client

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

async def main():
    import logging
    import traceback
    logging.basicConfig(level=logging.DEBUG)

    import mcp
    import pydantic
    from mcp.types import ListToolsRequest, ListToolsResult

    print(f"MCP Version: {mcp.__version__ if hasattr(mcp, '__version__') else 'unknown'}")
    print(f"Pydantic Version: {pydantic.VERSION}")
    try:
        print(f"ListToolsRequest Config: {ListToolsRequest.model_config}")
    except Exception as e:
        print(f"Could not print config: {e}")
    # print(f"ListToolsRequest Schema: {ListToolsRequest.model_json_schema()}")

    print("Initializing Client...")
    client = get_client()
    try:
        print("Calling list_tools...")
        tools = await client.list_tools()
        print(f"Success! Found {len(tools)} tools.")
        for t in tools:
            print(f"- {t.name}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
