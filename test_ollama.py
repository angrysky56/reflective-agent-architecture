
import asyncio
import logging
import os
import sys

# Add src to path
sys.path.append(os.path.abspath("/home/ty/Repositories/ai_workspace/reflective-agent-architecture"))

from src.compass.adapters import Message, RAALLMProvider

logging.basicConfig(level=logging.INFO)

async def test_ollama():
    print("Testing RAALLMProvider...")
    provider = RAALLMProvider()
    messages = [Message(role="user", content="Say hello")]

    print("Calling chat_completion...")
    full_response = ""
    async for chunk in provider.chat_completion(messages, stream=False):
        print(f"Chunk: {chunk!r}")
        full_response += chunk

    print(f"Full response: {full_response}")

if __name__ == "__main__":
    asyncio.run(test_ollama())
