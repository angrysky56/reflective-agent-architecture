"""
Adapters for integrating COMPASS with RAA infrastructure.
"""
import logging
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

@dataclass
class Message:
    """Message dataclass for LLM interaction."""
    role: str
    content: str

class RAALLMProvider:
    """
    Adapter for RAA's LLM (Ollama) to be used by COMPASS.
    """
    def __init__(self, model_name: Optional[str] = None):
        import os
        self.model_name = model_name or os.getenv("COMPASS_MODEL", "kimi-k2-thinking:cloud")
        self.dynamic_temperature_fn: Optional[Callable[[], float]] = None

    def set_dynamic_temperature_fn(self, fn: Callable[[], float]):
        """Set a callback to retrieve dynamic temperature based on cognitive state."""
        self.dynamic_temperature_fn = fn

    async def chat_completion(
        self,
        messages: List[Message],
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        tools: Optional[List[Dict]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Chat completion wrapper for Ollama using AsyncClient.
        """
        # Convert messages to Ollama format
        ollama_messages = [{"role": m.role, "content": m.content} for m in messages]

        # Determine temperature
        # If default (0.7) is passed, try to use dynamic temperature
        current_temp = temperature
        if abs(temperature - 0.7) < 0.01 and self.dynamic_temperature_fn:
            try:
                current_temp = self.dynamic_temperature_fn()
                logger.debug(f"Using dynamic temperature: {current_temp:.2f}")
            except Exception as e:
                logger.warning(f"Failed to get dynamic temperature: {e}")

        try:
            # Use Ollama's AsyncClient for proper async support
            from ollama import AsyncClient, ResponseError
            from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

            logger.info(f"RAALLMProvider: Creating AsyncClient for model {self.model_name}")
            client = AsyncClient()

            # If tools are provided, we should pass them (Ollama supports tools)
            options = {
                "temperature": current_temp,
                "num_predict": max_tokens
            }

            logger.info(f"RAALLMProvider: Calling client.chat with {len(ollama_messages)} messages, stream=True, tools={'provided' if tools else 'None'}")

            # Define retry strategy for 429 errors
            # Wait 2^x * 1 second between retries, up to 10 seconds max wait, for 5 attempts
            @retry(
                retry=retry_if_exception_type(ResponseError),
                wait=wait_exponential(multiplier=1, min=2, max=10),
                stop=stop_after_attempt(5),
                before_sleep=before_sleep_log(logger, logging.WARNING)
            )
            async def chat_with_retry():
                return await client.chat(
                    model=self.model_name,
                    messages=ollama_messages,
                    options=options,
                    tools=tools if tools else None,
                    stream=True
                )

            # Call Ollama with async client and retry logic
            try:
                response = await chat_with_retry()
            except ResponseError as e:
                if e.status_code == 429:
                    yield "Error: Rate limit exceeded (429) after retries. Please try again later."
                    return
                raise e

            logger.info("RAALLMProvider: Successfully received response generator")

            try:
                async for chunk in response:
                    logger.info(f"Raw chunk: {chunk}")
                    # Handle tool calls if present in chunk
                    # Ollama streaming response format:
                    # {'message': {'role': 'assistant', 'content': '...', 'tool_calls': [...]}}

                    # Handle chunk as object (Pydantic) or dict
                    msg = None
                    if hasattr(chunk, "message"):
                        msg = chunk.message
                    elif isinstance(chunk, dict):
                        msg = chunk.get("message", {})

                    content = ""
                    tool_calls = []

                    if msg:
                        if hasattr(msg, "content"):
                            content = msg.content
                        elif isinstance(msg, dict):
                            content = msg.get("content", "")

                        if hasattr(msg, "tool_calls"):
                            tool_calls = msg.tool_calls
                        elif isinstance(msg, dict):
                            tool_calls = msg.get("tool_calls", [])

                    if content:
                        yield content

                    if tool_calls:
                        # COMPASS expects a JSON string for tool calls in a specific format
                        # or it handles tool_calls object.
                        # IntegratedIntelligence._llm_intelligence handles:
                        # if chunk.strip().startswith('{"tool_calls":'):

                        # We need to serialize tool calls to JSON if we want to match that exact logic,
                        # OR we can modify IntegratedIntelligence to handle objects.
                        # For now, let's yield a JSON string representation
                        import json
                        # Ensure tool_calls are serializable (convert to dict if needed)
                        serializable_tool_calls = []
                        for tc in tool_calls:
                            if hasattr(tc, "model_dump"):
                                serializable_tool_calls.append(tc.model_dump())
                            elif hasattr(tc, "dict"):
                                serializable_tool_calls.append(tc.dict())
                            elif isinstance(tc, dict):
                                serializable_tool_calls.append(tc)
                            else:
                                # Fallback: try vars() or str()
                                try:
                                    serializable_tool_calls.append(vars(tc))
                                except Exception:
                                    serializable_tool_calls.append(str(tc))

                        yield json.dumps({"tool_calls": serializable_tool_calls})
            except ResponseError as e:
                if e.status_code == 429:
                    yield "Error: Rate limit exceeded (429) during streaming. Please try again later."
                    return
                raise e

        except Exception as e:
            # logger.error(f"Error in RAALLMProvider: {e}", exc_info=True)
            with open("/tmp/ollama_exception.log", "w") as f:
                import traceback
                traceback.print_exc(file=f)
            yield f"Error: {str(e)}"

    def _llm_generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Synchronous generation helper for simple text tasks.
        Uses ollama.chat directly.
        """
        import ollama
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": 0.7}
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"RAALLMProvider generate error: {e}")
            return f"Error generating text: {e}"

# Mock MCP Client for now, or implement if needed
class RAAMCPClient:
    """
    Adapter for RAA's tool system to look like an MCP client.
    """
    def __init__(self, workspace):
        self.workspace = workspace

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool on the workspace."""
        # This is a placeholder. RAA tools are methods on workspace or bridge.
        # We might need to map tool names to methods.
        logger.warning(f"Tool execution not fully implemented in adapter: {name}")
        return f"Tool {name} executed (mock)"

