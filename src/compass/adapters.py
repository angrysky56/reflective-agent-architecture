"""
Adapters for integrating COMPASS with RAA infrastructure.
"""

import logging
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, cast

from src.llm.factory import LLMFactory
from src.llm.provider import Message

logger = logging.getLogger(__name__)


class RAALLMProvider:
    """
    Adapter for RAA's LLM to be used by COMPASS.
    """

    def __init__(self, model_name: Optional[str] = None):
        import os

        self.model_name = model_name or os.getenv("COMPASS_MODEL", "google/gemini-3-pro-preview")
        self.provider_name = os.getenv("COMPASS_PROVIDER", os.getenv("LLM_PROVIDER", "openrouter"))
        # Use factory to get provider. COMPASS might use a different model/provider if configured.
        self.provider = LLMFactory.create_provider(
            provider_name=self.provider_name, model_name=self.model_name
        )
        self.dynamic_temperature_fn: Optional[Callable[[], float]] = None

    def set_dynamic_temperature_fn(self, fn: Callable[[], float]) -> None:
        """Set a callback to retrieve dynamic temperature based on cognitive state."""
        self.dynamic_temperature_fn = fn

    async def chat_completion(
        self,
        messages: List[Message],
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        tools: Optional[List[Dict]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Chat completion wrapper using the configured provider.
        """
        # Determine temperature
        current_temp = temperature
        if abs(temperature - 0.7) < 0.01 and self.dynamic_temperature_fn:
            try:
                current_temp = self.dynamic_temperature_fn()
                logger.debug(f"Using dynamic temperature: {current_temp:.2f}")
            except Exception as e:
                logger.warning(f"Failed to get dynamic temperature: {e}")

        try:
            async for chunk in self.provider.chat_completion(
                messages=messages,
                stream=stream,
                temperature=current_temp,
                max_tokens=max_tokens,
                tools=tools,
            ):
                yield chunk

        except Exception as e:
            logger.error(f"Error in RAALLMProvider: {e}", exc_info=True)
            yield f"Error: {str(e)}"

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 8000) -> str:
        """
        Synchronous generation wrapper.
        """
        return self.provider.generate(system_prompt, user_prompt, max_tokens=max_tokens)

    def _llm_generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Legacy helper, delegates to generate.
        """
        return self.generate(system_prompt, user_prompt)


# Mock MCP Client for now, or implement if needed
class RAAMCPClient:
    """
    Adapter for RAA's tool system to look like an MCP client.
    Wraps the RAAServer instance to expose get_available_tools and call_tool.
    """

    def __init__(self, server: Any):
        self.server = server

    def __repr__(self) -> str:
        return f"<RAAMCPClient server={type(self.server)}>"

    def get_available_tools(self) -> List[Any]:
        """Get available tools from the server."""
        if hasattr(self.server, "get_available_tools"):
            return cast(List[Any], self.server.get_available_tools())
        logger.warning(f"RAAMCPClient: Server {type(self.server)} has no get_available_tools")
        return []

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool on the server."""
        if hasattr(self.server, "call_tool"):
            return await self.server.call_tool(name, arguments)

        logger.error(f"RAAMCPClient: Server {type(self.server)} has no call_tool")
        raise AttributeError(f"Server {type(self.server)} has no call_tool")
        raise NotImplementedError(f"Server does not support call_tool: {type(self.server)}")
