import logging
import os
from typing import AsyncGenerator, Dict, List, Optional

from huggingface_hub import AsyncInferenceClient, InferenceClient
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from reflective_agent_architecture.llm.provider import BaseLLMProvider, Message

logger = logging.getLogger(__name__)


class HuggingFaceProvider(BaseLLMProvider):
    """Hugging Face implementation of LLM provider."""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("HF_TOKEN")
        if not self.api_key:
            logger.warning("Hugging Face token not found. Please set HF_TOKEN.")

    @retry(
        # HuggingFace Hub errors are often generic exceptions or requests exceptions
        # We'll retry on any Exception for now, or refine if we can import specific errors
        # Actually, let's try to be a bit safer and retry on Exception but log it
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 16000,
        tools: Optional[List[Dict]] = None,
    ) -> str:
        try:
            client = InferenceClient(token=self.api_key)
            # HF Inference API format can vary. Using chat completion style if supported, else text generation.
            # For simplicity, we assume a chat-compatible model or use a simple prompt structure.
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = client.chat_completion(
                messages=messages, model=self.model_name, max_tokens=max_tokens, temperature=0.7
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("HuggingFace API returned empty content")
            return str(content)
        except Exception:
            # We are retrying on generic Exception for HF, so we must re-raise it
            # But we also want to eventually return an error string if it fails after retries.
            # Tenacity will raise RetryError wrapping the original exception.
            # So we should just raise here.
            # Wait, if we raise here, tenacity catches it.
            # If tenacity gives up, it raises RetryError.
            # The caller of generate() (e.g. server.py) expects a string return usually?
            # Or does it handle exceptions?
            # server.py usually expects a string.
            # So we might need to wrap the whole thing or let it crash?
            # For now, let's re-raise to ensure retry works.
            raise
            # logger.error(f"HuggingFace generate error: {e}")
            # return f"Error generating text: {e}"

    @retry(
        retry=retry_if_exception_type(Exception),
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
        tools: Optional[List[Dict]] = None,
    ) -> AsyncGenerator[str, None]:

        client = AsyncInferenceClient(token=self.api_key)
        hf_messages = [{"role": m.role, "content": m.content} for m in messages]

        try:
            response = await client.chat_completion(
                messages=hf_messages,
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )

            async for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

        except Exception:
            raise
            # yield f"Error: {str(e)}"
