import logging
import os
from typing import AsyncGenerator, Dict, List, Optional

from anthropic import Anthropic, AsyncAnthropic

from src.llm.provider import BaseLLMProvider, Message

logger = logging.getLogger(__name__)

class AnthropicProvider(BaseLLMProvider):
    """Anthropic implementation of LLM provider."""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("Anthropic API key not found. Please set ANTHROPIC_API_KEY.")

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 1000) -> str:
        try:
            client = Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model_name,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic generate error: {e}")
            return f"Error generating text: {e}"

    async def chat_completion(
        self,
        messages: List[Message],
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        tools: Optional[List[Dict]] = None
    ) -> AsyncGenerator[str, None]:

        client = AsyncAnthropic(api_key=self.api_key)

        # Anthropic separates system prompt
        system_prompt = ""
        anthropic_messages = []
        for m in messages:
            if m.role == "system":
                system_prompt = m.content
            else:
                anthropic_messages.append({"role": m.role, "content": m.content})

        try:
            async with client.messages.stream(
                model=self.model_name,
                system=system_prompt,
                messages=anthropic_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                # tools=tools # Anthropic tool format differs, would need adapter
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            yield f"Error: {str(e)}"
