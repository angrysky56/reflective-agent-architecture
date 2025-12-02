import logging
import os
from typing import AsyncGenerator, Dict, List, Optional

from openai import AsyncOpenAI, OpenAI

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

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 1000) -> str:
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
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenRouter generate error: {e}")
            return f"Error generating text: {e}"

    async def chat_completion(
        self,
        messages: List[Message],
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        tools: Optional[List[Dict]] = None
    ) -> AsyncGenerator[str, None]:

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
                    # OpenRouter streams tool calls in parts similar to OpenAI
                    # For now, we'll use the same simplified approach as OpenAI provider
                    pass

        except Exception as e:
            yield f"Error: {str(e)}"
