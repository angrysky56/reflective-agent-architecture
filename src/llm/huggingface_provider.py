import logging
import os
from typing import AsyncGenerator, Dict, List, Optional

from huggingface_hub import AsyncInferenceClient, InferenceClient

from src.llm.provider import BaseLLMProvider, Message

logger = logging.getLogger(__name__)

class HuggingFaceProvider(BaseLLMProvider):
    """Hugging Face implementation of LLM provider."""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("HF_TOKEN")
        if not self.api_key:
            logger.warning("Hugging Face token not found. Please set HF_TOKEN.")

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 16000) -> str:
        try:
            client = InferenceClient(token=self.api_key)
            # HF Inference API format can vary. Using chat completion style if supported, else text generation.
            # For simplicity, we assume a chat-compatible model or use a simple prompt structure.
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            response = client.chat_completion(
                messages=messages,
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"HuggingFace generate error: {e}")
            return f"Error generating text: {e}"

    async def chat_completion(
        self,
        messages: List[Message],
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 16000,
        tools: Optional[List[Dict]] = None
    ) -> AsyncGenerator[str, None]:

        client = AsyncInferenceClient(token=self.api_key)
        hf_messages = [{"role": m.role, "content": m.content} for m in messages]

        try:
            response = await client.chat_completion(
                messages=hf_messages,
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )

            async for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

        except Exception as e:
            yield f"Error: {str(e)}"
