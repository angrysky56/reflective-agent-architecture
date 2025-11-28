import sys

import ollama

models = ["qwen3:4b", "kimi-k2-thinking:cloud"]

for model in models:
    print(f"\nTesting model: {model}")
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "user", "content": "Say hello."}
            ]
        )
        print("Response object keys:", response.keys())
        if 'message' in response:
            print("Content:", response['message']['content'])
        else:
            print("No message in response")
    except Exception as e:
        print(f"Error testing {model}: {e}")
