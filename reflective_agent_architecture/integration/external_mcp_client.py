import asyncio
import json
import logging
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool

logger = logging.getLogger(__name__)

class ExternalMCPManager:
    """
    Manages connections to external MCP servers defined in a configuration file.
    Aggregates tools from all connected servers.
    """

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = os.getenv("COMPASS_MCP_CONFIG", "compass_mcp_config.json")

        self.config_path = Path(config_path)
        self.stack = AsyncExitStack()
        self.sessions: Dict[str, ClientSession] = {}
        self.tools_map: Dict[str, Tuple[str, Tool]] = {}  # tool_name -> (server_name, Tool)
        self.is_initialized = False
        logger.info(f"ExternalMCPManager initialized with config path: {self.config_path}")

    async def initialize(self) -> None:
        """Initialize connections to all configured servers."""
        if self.is_initialized:
            logger.info("ExternalMCPManager already initialized, skipping.")
            return

        # Try multiple locations for config if it's relative
        search_paths = [
            self.config_path,
            Path(os.getcwd()) / self.config_path,
            Path(__file__).parent.parent.parent.parent / self.config_path, # Project root
            Path(__file__).parent.parent / self.config_path, # Package root
        ]

        found_path = None
        for p in search_paths:
            if p.exists():
                found_path = p
                break

        if not found_path:
            logger.warning(f"MCP config file not found in search paths: {[str(p) for p in search_paths]}")
            return

        logger.info(f"Loading MCP config from: {found_path}")
        try:
            with open(found_path, "r") as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")
            return

        mcp_servers = config.get("mcpServers", {})
        logger.info(f"Found {len(mcp_servers)} servers in config: {list(mcp_servers.keys())}")

        tasks = []
        for name, server_config in mcp_servers.items():
            # Skip entries that look like the current server to avoid infinite recursion
            if "reflective-agent-architecture" in name or "raa-server" in str(server_config.get("args", [])):
                logger.info(f"Skipping self-reference in external MCP config: {name}")
                continue
            tasks.append(self._connect_to_server(name, server_config))

        if not tasks:
            logger.info("No external MCP servers to connect to.")
            self.is_initialized = True
            return

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for name, result in zip(mcp_servers.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Failed to initialize server {name}: {result}")

        self.is_initialized = True
        logger.info("ExternalMCPManager initialization complete.")

    async def _connect_to_server(self, name: str, server_config: Dict[str, Any]) -> None:
        """Connect to a single MCP server."""
        command = server_config.get("command")
        args = server_config.get("args", [])
        env = server_config.get("env", {})

        full_env = os.environ.copy()
        full_env.update(env)

        try:
            logger.info(f"Attempting to connect to MCP server: {name}")
            server_params = StdioServerParameters(command=command, args=args, env=full_env)

            read, write = await self.stack.enter_async_context(stdio_client(server_params))
            session = await self.stack.enter_async_context(ClientSession(read, write))

            await asyncio.wait_for(session.initialize(), timeout=15.0)
            self.sessions[name] = session
            logger.info(f"Successfully connected to MCP server: {name}")

            await self.refresh_tools_for_server(name)

        except Exception as e:
            logger.error(f"Failed to connect to MCP server '{name}': {e}")

    async def refresh_tools_for_server(self, server_name: str) -> None:
        """Fetch and update tools for a specific server."""
        session = self.sessions.get(server_name)
        if not session:
            return

        try:
            result = await session.list_tools()
            tools = result.tools

            # Clear old tools from this server
            self.tools_map = {k: v for k, v in self.tools_map.items() if v[0] != server_name}

            for tool in tools:
                self.tools_map[tool.name] = (server_name, tool)

            logger.info(f"Loaded {len(tools)} tools from server '{server_name}'.")
        except Exception as e:
            logger.error(f"Error refreshing tools for {server_name}: {e}")

    async def cleanup(self) -> None:
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

        try:
            return await session.call_tool(name, arguments)
        except Exception as e:
            logger.error(f"Error calling tool '{name}' on server '{server_name}': {e}")
            raise RuntimeError(f"External tool call failed: {e}") from e
