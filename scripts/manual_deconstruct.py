import asyncio
import json
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv

from src.server import get_raa_context

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

async def main():
    print("Initializing RAA Context...")
    ctx = get_raa_context()
    # Ensure fully initialized
    if not ctx.is_initialized:
        ctx.initialize()

    # Override model to use local Qwen3 (though we will mock it)
    print("Switching to local model: qwen3")
    ctx.workspace.config.llm_model = "qwen3"

    # Mock LLM generation to bypass rate limits and speed up
    original_generate = ctx.workspace._llm_generate

    def mock_generate(system_prompt, user_prompt):
        print(f"DEBUG: Mocking LLM for prompt: {user_prompt[:50]}...")
        if "Ground of Being" in user_prompt:
            return json.dumps({
                "state_fragment": "Ontological Depth / The Abyss (Ungrund)",
                "agent_fragment": "The Unconditioned / The Absolute",
                "action_fragment": "Preceding and Sustaining Essence"
            })
        elif "Being as Such" in user_prompt:
            return json.dumps({
                "state_fragment": "Metaphysical Reality / Substance",
                "agent_fragment": "The Prime Mover / Actus Purus",
                "action_fragment": "Existing qua Existing (Fundamental Nature)"
            })
        return original_generate(system_prompt, user_prompt)

    ctx.workspace._llm_generate = mock_generate

    problems = [
        "The Ground of Being (Paul Tillich/Heidegger) - the absolute source of existence, preceding essence.",
        "Being as Such (Aristotle/Aquinas) - the fundamental nature of existence qua existence."
    ]

    for problem in problems:
        print(f"\nDeconstructing: {problem}")
        try:
            result = ctx.execute_deconstruct(problem)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error executing deconstruct for '{problem}': {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
