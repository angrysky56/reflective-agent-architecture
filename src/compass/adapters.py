"""
Adapters for integrating COMPASS with RAA infrastructure.
"""
import logging
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional

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

        try:
            # Use Ollama's AsyncClient for proper async support
            from ollama import AsyncClient

            logger.info(f"RAALLMProvider: Creating AsyncClient for model {self.model_name}")
            client = AsyncClient()

            # DEBUG: Print exact tools format being passed
            import json
            import sys

            # Force flush and write to file
            debug_msg = f"\n==== DEBUG: RAALLMProvider call at {__import__('datetime').datetime.now()} ====\n"
            debug_msg += f"Model: {self.model_name}\n"
            debug_msg += f"Messages count: {len(ollama_messages)}\n"
            debug_msg += f"Tools provided: {tools is not None}\n"
            if tools:
                debug_msg += f"Tools count: {len(tools)}\n"
            debug_msg += "="*50 + "\n"

            # Write to file AND stdout
            with open("/tmp/compass_llm_calls.log", "a") as f:
                f.write(debug_msg)
                f.flush()
            # print(debug_msg, flush=True)
            sys.stdout.flush()


            # If tools are provided, we should pass them (Ollama supports tools)
            options = {
                "temperature": temperature,
                "num_predict": max_tokens
            }

            logger.info(f"RAALLMProvider: Calling client.chat with {len(ollama_messages)} messages, stream=True, tools={'provided' if tools else 'None'}")
            # Call Ollama with async client
            response = await client.chat(
                model=self.model_name,
                messages=ollama_messages,
                options=options,
                tools=tools if tools else None,
                stream=True  # We stream to yield chunks
            )

            logger.info("RAALLMProvider: Successfully received response generator")

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

        except Exception as e:
            # logger.error(f"Error in RAALLMProvider: {e}", exc_info=True)
            with open("/tmp/ollama_exception.log", "w") as f:
                import traceback
                traceback.print_exc(file=f)
            yield f"Error: {str(e)}"

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

