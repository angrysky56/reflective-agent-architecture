import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

import ollama
from ollama import AsyncClient, ResponseError
from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.llm.provider import BaseLLMProvider, Message

logger = logging.getLogger(__name__)

class OllamaProvider(BaseLLMProvider):
    """Ollama implementation of LLM provider."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 16000) -> str:
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={"num_predict": max_tokens, "temperature": 0.7}
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama generate error: {e}")
            return f"Error generating text: {e}"

    async def chat_completion(
        self,
        messages: List[Message],
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 16000,
        tools: Optional[List[Dict]] = None
    ) -> AsyncGenerator[str, None]:

        ollama_messages = [{"role": m.role, "content": m.content} for m in messages]
        client = AsyncClient()

        options = {
            "temperature": temperature,
            "num_predict": max_tokens
        }

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

        try:
            response = await chat_with_retry()

            async for chunk in response:
                msg = chunk.get("message", {})
                content = msg.get("content", "")
                tool_calls = msg.get("tool_calls", [])

                if content:
                    yield content

                if tool_calls:
                    serializable_tool_calls = []
                    for tc in tool_calls:
                         # Ensure tool calls are dicts
                        if hasattr(tc, "model_dump"):
                            serializable_tool_calls.append(tc.model_dump())
                        elif isinstance(tc, dict):
                            serializable_tool_calls.append(tc)
                        else:
                             serializable_tool_calls.append(str(tc))
                    yield json.dumps({"tool_calls": serializable_tool_calls})

        except ResponseError as e:
            if e.status_code == 429:
                yield "Error: Rate limit exceeded (429). Please try again later."
            else:
                yield f"Error: {str(e)}"
        except Exception as e:
            yield f"Error: {str(e)}"
