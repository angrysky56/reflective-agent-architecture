import ollama

try:
    models = ollama.list()
    print("Available models:")
    for model in models['models']:
        print(f"- {model.model}")
except Exception as e:
    print(f"Error listing models: {e}")
