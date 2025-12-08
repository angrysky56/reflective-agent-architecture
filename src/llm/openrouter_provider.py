import logging
import os
from typing import AsyncGenerator, Dict, List, Optional

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, OpenAI, RateLimitError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.llm.provider import BaseLLMProvider, Message

logger = logging.getLogger(__name__)

class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter implementation of LLM provider using OpenAI-compatible API."""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"

        # OpenRouter optional headers for better tracking
        self.site_url = os.getenv("OPENROUTER_SITE_URL", "")
        self.app_name = os.getenv("OPENROUTER_APP_NAME", "reflective-agent-architecture")

        if not self.api_key:
            logger.warning("OpenRouter API key not found. Please set OPENROUTER_API_KEY.")

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 16000, tools: Optional[List[Dict]] = None) -> str:
        try:
            extra_headers = {}
            if self.site_url:
                extra_headers["HTTP-Referer"] = self.site_url
            if self.app_name:
                extra_headers["X-Title"] = self.app_name

            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers=extra_headers
            )

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.7,
                tools=tools if tools else None
            )

            if not response.choices:
                logger.error(f"OpenRouter returned no choices. Raw response: {response.model_dump_json()}")
                return f"Error: No choices returned. Raw: {response.model_dump_json()}"

            message = response.choices[0].message

            # 1. Handle Native Tool Calls (Convert to Dashboard's expected String format)
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                # Synthesize the markdown block expected by app.py
                import json
                try:
                    args = json.loads(tool_call.function.arguments)
                except Exception:
                    args = tool_call.function.arguments

                simulated_block = (
                    f"```json\n"
                    f"{json.dumps({'tool': tool_call.function.name, 'args': args})}\n"
                    f"```"
                )

                # Prepend content (thoughts) if it exists
                if message.content:
                    return f"{message.content}\n\n{simulated_block}"

                return simulated_block

            # 2. Handle Text Content
            return message.content or ""

        except (RateLimitError, APIConnectionError, APITimeoutError):
            raise  # Re-raise for tenacity to handle
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            logger.error(f"OpenRouter generate error: {e}\n{trace}")
            return f"Error generating text: {e}\n\nTraceback:\n{trace}"

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def chat_completion(
        self,
        messages: List[Message],
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 16000,
        tools: Optional[List[Dict]] = None
    ) -> AsyncGenerator[str, None]:
        if tools:
            logger.debug(f"OpenRouterProvider: Received {len(tools)} tools for chat completion.")
            # logger.debug(f"Tool names: {[t['function']['name'] for t in tools]}")
        else:
            logger.debug("OpenRouterProvider: No tools provided for chat completion.")

        extra_headers = {}
        if self.site_url:
            extra_headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            extra_headers["X-Title"] = self.app_name

        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers=extra_headers
        )

        openai_messages = [{"role": m.role, "content": m.content} for m in messages]

        try:
            # Respect the stream parameter
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools if tools else None,
                stream=stream
            )

            if stream:
                async for chunk in response:
                    delta = chunk.choices[0].delta
                    content = delta.content
                    tool_calls = delta.tool_calls

                    if content:
                        yield content

                    # TODO: Implement proper streaming tool call aggregation if needed later
                    # Currently we rely on non-streaming for robust tool usage
            else:
                # Non-streaming response
                message = response.choices[0].message
                content = message.content
                tool_calls = message.tool_calls

                if content:
                    yield content

                if tool_calls:
                    # Serialize tool calls to the format IntegratedIntelligence expects
                    import json
                    # Convert OpenAI ToolCall objects to dicts
                    tool_calls_dict = []
                    for tc in tool_calls:
                        tool_calls_dict.append({
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        })

                    yield json.dumps({"tool_calls": tool_calls_dict})

        except (RateLimitError, APIConnectionError, APITimeoutError):
            raise
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            logger.error(f"OpenRouter chat_completion error: {e}\n{trace}")
            yield f"Error: {str(e)}"
