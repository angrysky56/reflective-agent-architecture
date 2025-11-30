
"""
MCP Tool Adapter for COMPASS (RAA Integration).
"""
import json
from typing import Any, Dict, List, Tuple


def _validate_schema(schema: Dict) -> bool:
    """
    Validate that a tool schema is compatible with Ollama/OpenAI.
    """
    try:
        # 1. Must be an object type
        if schema.get("type") != "object":
            return False

        # 2. Must have properties dict (can be empty)
        if not isinstance(schema.get("properties", {}), dict):
            return False

        # 3. Check for specific problematic patterns
        # Some MCP tools might use "anyOf" or "oneOf" at top level which Ollama struggles with
        if "anyOf" in schema or "oneOf" in schema:
            # Simple check - if it's complex, skip for now to be safe
            return False

        return True
    except Exception:
        return False

async def get_available_tools_for_llm(mcp_client: Any) -> List[Dict]:
    """
    Get available tools in a format suitable for LLM calling.

    Args:
        mcp_client: The MCP client (RAAServerContext)

    Returns:
        List of tool definitions in OpenAI/Ollama format
    """
    if not mcp_client:
        return []

    try:
        # Get tools from client
        # mcp_client is RAAServerContext which has get_available_tools()
        mcp_tools = mcp_client.get_available_tools()

        llm_tools = []
        for tool in mcp_tools:
            # Validate schema first
            if not _validate_schema(tool.inputSchema):
                # Skipping tool invalid_tool: Invalid or complex schema
                continue

            # Convert MCP Tool to OpenAI format
            # MCP Tool has: name, description, inputSchema
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
        # Fallback or log error
        # Error fetching tools: {e}
        return []

def format_tool_call_for_mcp(tool_call: Dict) -> Tuple[str, Dict]:
    """
    Format an LLM tool call into (name, args) for MCP execution.
    """
    function = tool_call.get("function", {})
    name = function.get("name", "")
    try:
        args_str = function.get("arguments", "{}")
        if isinstance(args_str, str):
            args = json.loads(args_str)
        else:
            args = args_str
    except:
        args = {}
    return name, args
