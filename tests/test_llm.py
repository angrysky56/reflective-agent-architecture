import sys

import ollama

try:
    print("Testing Ollama connection...")
    response = ollama.chat(
        model="kimi-k2-thinking:cloud",
        messages=[{"role": "user", "content": "Hello, are you working?"}],
        options={"num_predict": 100, "temperature": 1.0}
    )
    print("Response received:")
    print(response)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
