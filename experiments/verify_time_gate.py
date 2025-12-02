
import asyncio
import json
import math
import os
import random
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.server import call_tool, get_raa_context


async def main():
    print("Initializing RAA Context...")
    ctx = get_raa_context()

    # Generate Alien Box Data (tanh(tanh(y)) + sin(x*z))
    print("Generating Alien Box Data...")
    data_points = []
    for _ in range(20):
        x = random.uniform(-5, 5)
        y = random.uniform(-5, 5)
        z = random.uniform(-5, 5)
        # Complex non-linear formula
        result = math.tanh(math.tanh(y)) + math.sin(x * z)
        data_points.append({"x": x, "y": y, "z": z, "result": result})

    # Format data for prompt
    data_str = "\n".join([f"x={d['x']:.2f}, y={d['y']:.2f}, z={d['z']:.2f} -> result={d['result']:.2f}" for d in data_points[:10]])
    task = f"Find the mathematical formula that governs this data:\n{data_str}\n... (and 10 more points). It seems complex."

    context = {
        "data_points": data_points,
        "hint": "It might involve tanh and sin.",
        "force_time_gate": True
    }

    print(f"Calling consult_compass with task: {task}")
    print("Expecting 'Time Gate' to trigger due to complexity...")

    try:
        # We expect the initial LLM attempt to be low confidence (or we hope so)
        # If it's high confidence, the Time Gate won't trigger.
        # But for this test, we want to see if the logic *can* run.
        # Note: The real test is if the logs show "High Entropy detected".

        result = await call_tool("consult_compass", {"task": task, "context": context})

        print("\nResult:")
        for content in result:
            print(content.text)

        if "[System 2 Intervention]" in str(result):
            print("\nVERIFICATION PASSED: Time Gate triggered and System 2 intervened.")
        else:
            print("\nVERIFICATION UNCERTAIN: Time Gate might not have triggered (maybe confidence was too high?). Check logs.")

    except Exception as e:
        print(f"\nVERIFICATION FAILED: Exception occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
