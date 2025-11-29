
"""
MCP Tool Adapter for COMPASS (RAA Integration).
"""
import json
from typing import Any, Dict, List, Tuple


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
        print(f"Error fetching tools: {e}")
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
