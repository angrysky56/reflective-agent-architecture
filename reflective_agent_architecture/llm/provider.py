from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, List, Optional


@dataclass
class Message:
    role: str
    content: str


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 16000,
        tools: Optional[List[Dict]] = None,
    ) -> str:
        """Synchronous generation for simple text tasks."""
        pass

    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Message],
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 16000,
        tools: Optional[List[Dict]] = None,
    ) -> AsyncGenerator[str, None]:
        """Async chat completion with streaming support."""
        raise NotImplementedError
        yield ""
