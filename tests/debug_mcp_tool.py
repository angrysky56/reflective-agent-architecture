
import asyncio
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger("debug_mcp")

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.integration.external_mcp_client import ExternalMCPManager


async def main():
    config_path = Path("compass_mcp_config.json").absolute()
    print(f"Config path: {config_path}", flush=True)

    manager = ExternalMCPManager(str(config_path))

    try:
        print("Initializing MCP Manager...", flush=True)
        await manager.initialize()

        print("\nAvailable Tools:", flush=True)
        tools = manager.get_tools()
        for t in tools:
            print(f"- {t.name}", flush=True)

        print("\nTesting brave_web_search...", flush=True)
        if "brave_web_search" in manager.tools_map:
            print("Calling brave_web_search with query 'Ethereum price'...", flush=True)
            try:
                result = await asyncio.wait_for(
                    manager.call_tool("brave_web_search", {"query": "Ethereum price USD"}),
                    timeout=30.0
                )
                print(f"Result: {result}", flush=True)
            except asyncio.TimeoutError:
                print("TIMEOUT: Call took longer than 30s", flush=True)
            except Exception as e:
                print(f"ERROR calling tool: {e}", flush=True)
        else:
            print("brave_web_search NOT FOUND in tools map", flush=True)

    finally:
        try:
            await manager.cleanup()
        except Exception as e:
            print(f"Cleanup error ignored: {e}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
