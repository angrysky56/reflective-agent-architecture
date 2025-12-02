import asyncio
import logging
import os
import sys

# Add src to path
sys.path.append(os.getcwd())

# Mock environment variables if needed
# os.environ["NEO4J_URI"] = "bolt://localhost:7687"
# os.environ["NEO4J_USER"] = "neo4j"
# os.environ["NEO4J_PASSWORD"] = "password"

# Import server
try:
    from src.server import RAA_TOOLS, list_tools
    print(f"RAA_TOOLS length: {len(RAA_TOOLS)}")

    found = False
    for tool in RAA_TOOLS:
        if tool.name == "evolve_formula":
            print("Found 'evolve_formula' in RAA_TOOLS!")
            print(f"Input Schema: {tool.inputSchema}")
            found = True
            break

    if not found:
        print("ERROR: 'evolve_formula' NOT found in RAA_TOOLS.")

    # Check list_tools
    print("\nChecking list_tools()...")
    # list_tools is async
    async def check():
        tools = await list_tools()
        names = [t.name for t in tools]
        print(f"Tools returned by list_tools: {names}")
        if "evolve_formula" in names:
            print("SUCCESS: 'evolve_formula' is returned by list_tools()")
        else:
            print("FAILURE: 'evolve_formula' is NOT returned by list_tools()")

    asyncio.run(check())

except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
