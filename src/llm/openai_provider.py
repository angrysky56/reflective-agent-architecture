import json
import logging
import os
from typing import AsyncGenerator, Dict, List, Optional

from openai import AsyncOpenAI, OpenAI

from src.llm.provider import BaseLLMProvider, Message

logger = logging.getLogger(__name__)

class OpenAIProvider(BaseLLMProvider):
    """OpenAI implementation of LLM provider (also works for LM Studio)."""

    def __init__(self, model_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")

        if not self.api_key and not self.base_url:
             # LM Studio might not need a key, but OpenAI does.
             # If base_url is set (e.g. localhost for LM Studio), we can default key to "lm-studio".
             if self.base_url:
                 self.api_key = "lm-studio"
             else:
                 logger.warning("OpenAI API key not found. Please set OPENAI_API_KEY.")

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 16000) -> str:
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generate error: {e}")
            return f"Error generating text: {e}"

    async def chat_completion(
        self,
        messages: List[Message],
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 16000,
        tools: Optional[List[Dict]] = None
    ) -> AsyncGenerator[str, None]:

        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        openai_messages = [{"role": m.role, "content": m.content} for m in messages]

        try:
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools if tools else None,
                stream=True
            )

            async for chunk in response:
                delta = chunk.choices[0].delta
                content = delta.content
                tool_calls = delta.tool_calls

                if content:
                    yield content

                if tool_calls:
                    # OpenAI streams tool calls in parts, which is complex to handle directly here
                    # without a full accumulator. For simplicity in this MVP, we might need to
                    # reconsider streaming tool calls or accumulate them.
                    # However, for RAA's current usage, we mostly need text streaming.
                    # If tool calls are critical in streaming, we'd need a more robust accumulator.
                    # For now, we'll yield a simplified representation or skip if partial.
                    pass

        except Exception as e:
            yield f"Error: {str(e)}"
