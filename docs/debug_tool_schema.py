
import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List


# Mock classes
@dataclass
class Tool:
    name: str
    description: str
    inputSchema: Dict

class MockMCPClient:
    def get_available_tools(self):
        return [
            Tool(
                name="read_file",
                description="Read a file from the filesystem",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file"}
                    },
                    "required": ["path"]
                }
            )
        ]

# Import function to test (copied for simplicity)
def _validate_schema(schema: Dict) -> bool:
    try:
        if schema.get("type") != "object": return False
        if not isinstance(schema.get("properties", {}), dict): return False
        if "anyOf" in schema or "oneOf" in schema: return False
        return True
    except Exception:
        return False

async def get_available_tools_for_llm(mcp_client: Any) -> List[Dict]:
    if not mcp_client: return []
    try:
        mcp_tools = mcp_client.get_available_tools()
        llm_tools = []
        for tool in mcp_tools:
            if not _validate_schema(tool.inputSchema): continue
            llm_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            }
            llm_tools.append(llm_tool)
        return llm_tools
    except Exception as e:
        print(f"Error: {e}")
        return []

async def main():
    client = MockMCPClient()
    tools = await get_available_tools_for_llm(client)
    print(json.dumps(tools, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
