
import asyncio
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
        logger.info(f"ExternalMCPManager initialized with config path: {config_path}")
        self.is_initialized = False

    async def initialize(self):
        """Initialize connections to all configured servers."""
        if self.is_initialized:
            logger.info("ExternalMCPManager already initialized, skipping.")
            return

        logger.info("Initializing ExternalMCPManager...")

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
        logger.info(f"debug: found {len(mcp_servers)} servers in config: {list(mcp_servers.keys())}")

        tasks = []
        for name, server_config in mcp_servers.items():
            # Skip self if present
            if name == "cognitive-workspace-db":
                continue
            tasks.append(self._connect_to_server(name, server_config))

        # Run connections with timeout and error handling
        # 10 second timeout for initialization to prevent hanging
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for name, result in zip(mcp_servers.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Failed to initialize server {name}: {result}")
            elif name == "cognitive-workspace-db":
                pass # Skipped above
            else:
                # Connection successful logic is handled inside _connect_to_server
                pass

        self.is_initialized = True
        logger.info("ExternalMCPManager initialization complete.")

    async def _connect_to_server(self, name: str, server_config: Dict[str, Any]):
        """Connect to a single MCP server."""
        command = server_config.get("command")
        args = server_config.get("args", [])
        env = server_config.get("env", {})

        # Merge with current env
        full_env = os.environ.copy()
        full_env.update(env)

        try:
            logger.info(f"Attempting to connect to MCP server: {name}")

            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=full_env
            )

            # Enter contexts
            read, write = await self.stack.enter_async_context(stdio_client(server_params))
            session = await self.stack.enter_async_context(ClientSession(read, write))

            # Initialize with timeout
            await asyncio.wait_for(session.initialize(), timeout=10.0)
            self.sessions[name] = session
            logger.info(f"Successfully connected to MCP server: {name}")

            # List tools
            logger.info(f"Fetching tools for server: {name}...")
            result = await session.list_tools()
            tools = result.tools
            logger.info(f"Server '{name}' returned {len(tools)} tools: {[t.name for t in tools]}")

            for tool in tools:
                if tool.name in self.tools_map:
                    logger.warning(f"Duplicate tool name '{tool.name}' from server '{name}'. Overwriting existing tool from '{self.tools_map[tool.name][0]}'.")
                self.tools_map[tool.name] = (name, tool)

            logger.info(f"Loaded {len(tools)} tools from server '{name}'.")

        except Exception as e:
            logger.error(f"Failed to connect to MCP server '{name}': {e}", exc_info=True)

    async def cleanup(self):
        """Close all connections."""
        logger.info("Closing external MCP connections...")
        await self.stack.aclose()
        self.sessions.clear()
        self.tools_map.clear()
        self.is_initialized = False

    def get_tools(self) -> List[Tool]:
        """Get all available tools from external servers."""
        tools = [tool for _, tool in self.tools_map.values()]
        logger.debug(f"ExternalMCPManager.get_tools returning {len(tools)} tools: {[t.name for t in tools]}")
        return tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the appropriate server."""
        if name not in self.tools_map:
            raise ValueError(f"Tool '{name}' not found in external servers")

        server_name, _ = self.tools_map[name]
        session = self.sessions.get(server_name)

        if not session:
            raise RuntimeError(f"Session for server '{server_name}' is not active")

        logger.info(f"Calling external tool '{name}' on server '{server_name}'")
        try:
            result = await session.call_tool(name, arguments)
            return result
        except Exception as e:
            logger.error(f"Error calling tool '{name}' on server '{server_name}': {e}")
            raise RuntimeError(f"External tool call failed: {e}") from e
