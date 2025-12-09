import os
import sys

from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Load env
load_dotenv()

from src.llm.openrouter_provider import OpenRouterProvider


def test_openrouter():
    print("Testing OpenRouterProvider...")
    base_url = "https://openrouter.ai/api/v1"
    key = os.getenv("OPENROUTER_API_KEY")
    print(f"Key present: {bool(key)}")

    provider = OpenRouterProvider(
        model_name="deepcogito/cogito-v2-preview-llama-405b",
        api_key=key
    )

    tools = [{
        "type": "function",
        "function": {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {
                "type": "object",
                "properties": {"arg": {"type": "string"}}
            }
        }
    }]

    try:
        response = provider.generate(
            system_prompt="You are a helpful assistant.",
            user_prompt="Hello, world!",
            tools=tools
        )
        print("\n--- RESPONSE ---")
        print(response)
        print("----------------")
    except Exception as e:
        print(f"\nCaught Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_openrouter()
