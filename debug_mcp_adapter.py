
import json
import asyncio
from typing import Dict, Any, List
from unittest.mock import MagicMock

# Import the functions to test
# We need to mock the imports or copy the functions here since we can't easily import from src without setting up path
# For simplicity, I will copy the relevant functions from mcp_tool_adapter.py

def format_tool_call_for_mcp(tool_call: Dict):
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
    except Exception as e:
        print(f"Error parsing arguments: {e}")
        args = {}
    return name, args

# Test cases
def test_format_tool_call():
    print("Testing format_tool_call_for_mcp...")

    # Case 1: Standard OpenAI format with JSON string arguments
    tool_call_1 = {
        "id": "call_123",
        "type": "function",
        "function": {
            "name": "read_file",
            "arguments": "{\"path\": \"/tmp/test.txt\"}"
        }
    }
    name, args = format_tool_call_for_mcp(tool_call_1)
    print(f"Case 1: name={name}, args={args}")
    assert name == "read_file"
    assert args == {"path": "/tmp/test.txt"}

    # Case 2: Arguments as dict (already parsed)
    tool_call_2 = {
        "function": {
            "name": "read_file",
            "arguments": {"path": "/tmp/test.txt"}
        }
    }
    name, args = format_tool_call_for_mcp(tool_call_2)
    print(f"Case 2: name={name}, args={args}")
    assert args == {"path": "/tmp/test.txt"}

    # Case 3: Malformed JSON
    tool_call_3 = {
        "function": {
            "name": "read_file",
            "arguments": "{path: /tmp/test.txt}" # Invalid JSON
        }
    }
    name, args = format_tool_call_for_mcp(tool_call_3)
    print(f"Case 3: name={name}, args={args}")
    # Should return empty dict due to exception handling
    assert args == {}

    print("format_tool_call_for_mcp tests passed!")

if __name__ == "__main__":
    test_format_tool_call()
