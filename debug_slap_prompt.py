
import asyncio
import logging
import os
import sys

# Add src to path
sys.path.append(os.path.abspath("/home/ty/Repositories/ai_workspace/reflective-agent-architecture"))

from src.compass.adapters import Message, RAALLMProvider

logging.basicConfig(level=logging.INFO)

async def debug_slap():
    print("Testing SLAP Prompt with direct ollama...")
    import ollama

    system_prompt = "You are the SLAP engine." # Shortened for brevity in test
    user_prompt = "Task: 2+2. Output JSON."

    messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]

    print("Calling ollama.chat...")
    response = ollama.chat(model='kimi-k2-thinking:cloud', messages=messages, stream=True)

    full_response = ""
    for chunk in response:
        print(f"Chunk type: {type(chunk)}")
        print(f"Chunk dir: {dir(chunk)}")
        print(f"Chunk raw: {chunk}")

        if hasattr(chunk, 'message'):
            msg = chunk.message
            print(f"Message content: {msg.content!r}")
            full_response += msg.content
        elif isinstance(chunk, dict):
             msg = chunk.get('message', {})
             print(f"Message content: {msg.get('content', '')!r}")
             full_response += msg.get('content', '')

    print(f"Full response: {full_response}")

if __name__ == "__main__":
    asyncio.run(debug_slap())
