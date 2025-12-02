import os
import sys

import ollama
from dotenv import load_dotenv

load_dotenv()

model = os.getenv("LLM_MODEL", "qwen3-vl:235b-instruct-cloud")
print(f"Testing Ollama connection with model: {model}")

try:
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": "Hello, are you working?"}],
        options={"num_predict": 100, "temperature": 1.0}
    )
    print("Response received:")
    print(response)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
