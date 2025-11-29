
import json
import logging
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, Tuple

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool

logger = logging.getLogger(__name__)

class ExternalMCPManager:
    """
    Manages connections to external MCP servers defined in a configuration file.
    Aggregates tools from all connected servers.
    """
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.stack = AsyncExitStack()
        self.sessions: Dict[str, ClientSession] = {}
        self.tools_map: Dict[str, Tuple[str, Tool]] = {} # tool_name -> (server_name, Tool)
        self.is_initialized = False

    async def initialize(self):
        """Initialize connections to all configured servers."""
        if self.is_initialized:
            return

        if not self.config_path.exists():
            logger.warning(f"MCP config file not found: {self.config_path}")
            return

        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")
            return

        mcp_servers = config.get("mcpServers", {})

        for name, server_config in mcp_servers.items():
            # Skip self if present
            if name == "cognitive-workspace-db":
                continue

            command = server_config.get("command")
            args = server_config.get("args", [])
            env = server_config.get("env", {})

            # Merge with current env
            full_env = os.environ.copy()
            full_env.update(env)

            try:
                logger.info(f"Connecting to MCP server: {name}")

                server_params = StdioServerParameters(
                    command=command,
                    args=args,
                    env=full_env
                )

                # Enter contexts
                read, write = await self.stack.enter_async_context(stdio_client(server_params))
                session = await self.stack.enter_async_context(ClientSession(read, write))

                await session.initialize()

                # List tools
                result = await session.list_tools()
                tools = result.tools

                self.sessions[name] = session

                for tool in tools:
                    if tool.name in self.tools_map:
                        logger.warning(f"Duplicate tool name '{tool.name}' from server '{name}'. Overwriting.")
                    self.tools_map[tool.name] = (name, tool)

                logger.info(f"Connected to {name}, loaded {len(tools)} tools")

            except Exception as e:
                logger.error(f"Failed to connect to MCP server {name}: {e}")

        self.is_initialized = True

    async def cleanup(self):
        """Close all connections."""
        logger.info("Closing external MCP connections...")
        await self.stack.aclose()
        self.sessions.clear()
        self.tools_map.clear()
        self.is_initialized = False

    def get_tools(self) -> List[Tool]:
        """Get all available tools from external servers."""
        return [tool for _, tool in self.tools_map.values()]

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the appropriate server."""
        if name not in self.tools_map:
            raise ValueError(f"Tool '{name}' not found in external servers")

        server_name, _ = self.tools_map[name]
        session = self.sessions.get(server_name)

        if not session:
            raise RuntimeError(f"Session for server '{server_name}' is not active")

        logger.info(f"Calling external tool '{name}' on server '{server_name}'")
        result = await session.call_tool(name, arguments)
        return result
