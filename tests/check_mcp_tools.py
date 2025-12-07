"""
Quick diagnostic: Check what tools the MCP server is actually advertising.

This will help us understand if the server is properly exposing evolve_formula.
"""

import json
import subprocess

# Use the MCP inspector to list tools
result = subprocess.run(
    ["mcp", "inspect", "tools"],
    capture_output=True,
    text=True,
    cwd="/home/ty/Repositories/ai_workspace/reflective-agent-architecture"
)

print("=== MCP Tools List ===")
print(result.stdout)
print(result.stderr)

# Check for evolve_formula
if "evolve_formula" in result.stdout:
    print("\n✅ SUCCESS: evolve_formula is visible to MCP client!")
else:
    print("\n❌ FAILURE: evolve_formula is NOT visible to MCP client")
    print("\nThis suggests the server process hasn't fully restarted or there's a caching issue.")
