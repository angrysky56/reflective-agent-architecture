import json
import logging
import os
from typing import AsyncGenerator, Dict, List, Optional

import google.generativeai as genai
from google.generativeai.types import HarmBlockThreshold, HarmCategory

from src.llm.provider import BaseLLMProvider, Message

logger = logging.getLogger(__name__)

class GeminiProvider(BaseLLMProvider):
    """Google Gemini implementation of LLM provider."""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("Gemini API key not found. Please set GEMINI_API_KEY.")
        else:
            genai.configure(api_key=self.api_key)

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 1000) -> str:
        try:
            model = genai.GenerativeModel(self.model_name)
            # Gemini doesn't have a strict "system" role in the same way, but we can prepend it.
            # Or use system_instruction if supported by the specific model version.
            # For broad compatibility, we'll prepend.
            full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"

            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.7
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini generate error: {e}")
            return f"Error generating text: {e}"

    async def chat_completion(
        self,
        messages: List[Message],
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        tools: Optional[List[Dict]] = None
    ) -> AsyncGenerator[str, None]:

        try:
            model = genai.GenerativeModel(self.model_name)

            # Convert messages to Gemini history format
            history = []
            last_user_msg = ""

            for m in messages[:-1]: # All but last
                role = "user" if m.role == "user" else "model"
                history.append({"role": role, "parts": [m.content]})

            if messages:
                last_user_msg = messages[-1].content

            chat = model.start_chat(history=history)

            # Note: Gemini async streaming support in the python lib might vary.
            # We'll use the async generator if available, otherwise wrap sync.
            response = await chat.send_message_async(
                last_user_msg,
                stream=True,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature
                )
            )

            async for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            yield f"Error: {str(e)}"
