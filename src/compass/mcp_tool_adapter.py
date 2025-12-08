
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
        return True

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
    import logging
    logger = logging.getLogger("IntegratedIntelligence")



    if not mcp_client:
        logger.warning("MCP Tool Adapter: mcp_client is None")

        return []

    try:
        # Get tools from client
        # mcp_client is RAAServerContext which has get_available_tools()
        logger.info(f"MCP Tool Adapter: Calling get_available_tools() on {type(mcp_client).__name__}")


        mcp_tools = mcp_client.get_available_tools()
        logger.info(f"MCP Tool Adapter: Got {len(mcp_tools)} total tools from client")



        # Check if external_mcp is initialized
        if hasattr(mcp_client, 'external_mcp') and mcp_client.external_mcp:
            logger.info(f"MCP Tool Adapter: external_mcp is_initialized={mcp_client.external_mcp.is_initialized}")
            logger.info(f"MCP Tool Adapter: external_mcp tools_map has {len(mcp_client.external_mcp.tools_map)} tools")


        else:
            logger.warning("MCP Tool Adapter: external_mcp not available or is None on client")


        llm_tools = []
        for idx, tool in enumerate(mcp_tools):
            logger.debug(f"MCP Tool Adapter: Processing tool {idx}: {tool.name}")

            # Validate schema first
            if not _validate_schema(tool.inputSchema):
                logger.warning(f"Skipping tool {tool.name}: Invalid schema")

                continue

            # Sanitize schema for Ollama/OpenAI compatibility
            # Create a deep copy to avoid modifying original
            import copy
            schema = copy.deepcopy(tool.inputSchema)

            # Helper to recursively sanitize
            def sanitize_node(node):
                if isinstance(node, dict):
                    # Remove 'default', 'additionalProperties', '$schema'
                    node.pop("default", None)
                    node.pop("additionalProperties", None)
                    node.pop("$schema", None)

                    # Handle complex types (anyOf, oneOf, allOf)
                    # Simplify to the first option or string to avoid Ollama errors
                    for complex_key in ["anyOf", "oneOf", "allOf"]:
                        if complex_key in node:
                            options = node.pop(complex_key)
                            if isinstance(options, list) and len(options) > 0:
                                # Take the first option's properties and merge them
                                # This is a simplification but keeps the schema valid for Ollama
                                first_option = options[0]
                                if isinstance(first_option, dict):
                                    # Recursively sanitize the option first
                                    sanitize_node(first_option)
                                    node.update(first_option)
                            else:
                                # Fallback to string if options are empty/invalid
                                node["type"] = "string"

                    for key, value in node.items():
                        sanitize_node(value)
                elif isinstance(node, list):
                    for item in node:
                        sanitize_node(item)

            sanitize_node(schema)

            # Convert MCP Tool to OpenAI format
            llm_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": schema
                }
            }
            llm_tools.append(llm_tool)
            logger.debug(f"MCP Tool Adapter: Converted tool: {tool.name}")

        logger.info(f"MCP Tool Adapter: Successfully converted {len(llm_tools)} tools for LLM")


        return llm_tools

    except Exception as e:
        logger.error(f"MCP Tool Adapter: Error fetching tools: {e}", exc_info=True)

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
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                # Fallback for single-quoted JSON or Python-like dict string
                import ast
                try:
                    args = ast.literal_eval(args_str)
                    if not isinstance(args, dict):
                        args = {}
                except Exception:
                    args = {}
        elif isinstance(args_str, dict):
            args = args_str
        else:
            args = {}

    except Exception:
        args = {}
    return name, args
