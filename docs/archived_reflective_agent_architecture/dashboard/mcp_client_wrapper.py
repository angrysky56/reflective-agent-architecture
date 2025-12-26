import json
import os
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, cast

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class RAAClient:
    """
    A wrapper around the MCP Client to communicate with the RAA Server.
    Manages the stdio connection subprocess using environment variables for configuration.
    """

    def __init__(self, config_path: str = "src/dashboard/internal_bridge_config.json"):
        self.client = None
        self.exit_stack = AsyncExitStack()
        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load MCP server configuration from json file."""
        # Resolve path relative to CWD if needed, or absolute
        if not os.path.isabs(self.config_path):
            config_path = os.path.join(os.getcwd(), self.config_path)
        else:
            config_path = self.config_path

        try:
            with open(config_path, "r") as f:
                data = json.load(f)
                return cast(Dict[str, Any], data.get("mcpServers", {}).get("raa-server", {}))
        except Exception as e:
            print(f"Failed to load MCP config from {config_path}: {e}")
            # Fallback defaults
            return {
                "command": "uv",
                "args": ["run", "-q", "src/server.py"],
                "env": {"PYTHONPATH": "."},
            }

    @asynccontextmanager
    async def session(self) -> AsyncIterator[ClientSession]:
        """
        Yields an active MCP ClientSession.
        """
        # Prepare environment
        env = os.environ.copy()
        config_env = self._config.get("env", {})

        # Merge config env
        for k, v in config_env.items():
            env[k] = v

        # Ensure PYTHONPATH resolves correctly (if . is used)
        if env.get("PYTHONPATH") == ".":
            env["PYTHONPATH"] = os.getcwd()

        server_params = StdioServerParameters(
            command=self._config.get("command", "uv"),
            args=self._config.get("args", ["run", "-q", "src/server.py"]),
            env=env,
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the server."""
        async with self.session() as session:
            result = await session.list_tools()
            return [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema,
                }
                for tool in result.tools
            ]

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call a specific tool on the server."""
        async with self.session() as session:
            result = await session.call_tool(tool_name, arguments)
            text_output = ""
            if hasattr(result, "content"):
                for item in result.content:
                    if hasattr(item, "text"):
                        text_output += item.text + "\n"
            return text_output.strip()


# Singleton-like helper for Streamlit
def get_client() -> RAAClient:
    return RAAClient()
