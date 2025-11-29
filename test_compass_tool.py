
import asyncio
import logging
import os
import sys
from typing import Any, Dict

# Add src to path
sys.path.append(os.path.abspath("/home/ty/Repositories/ai_workspace/reflective-agent-architecture"))

from src.director.director_core import DirectorMVP
from src.server import RAAServerContext, get_raa_context

logging.basicConfig(level=logging.INFO)

async def test_consult_compass():
    print("Initializing RAA Context...")
    # We need to mock the workspace or initialize it properly
    # Initialization might take time and require DB connections.
    # Instead, let's manually initialize Director and COMPASS.

    from src.compass.adapters import RAALLMProvider
    from src.compass.compass_framework import COMPASS

    print("Initializing COMPASS...")
    compass = COMPASS(llm_provider=RAALLMProvider())

    task = "What is 2 + 2?"
    context = {}

    print(f"Processing task: {task}")
    result = await compass.process_task(task, context, max_iterations=1)

    print("Result received!")
    import json
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(test_consult_compass())
