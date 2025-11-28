import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.integration.agent_factory import AgentFactory


# Mock LLM generator for the demo
def mock_llm_generate(system_prompt, user_prompt):
    print(f"\n[LLM Input] System: {system_prompt[:50]}... | User: {user_prompt[:50]}...")

    if "Create a System Prompt" in user_prompt:
        return (
            "You are a Debater Agent. Your goal is to arbitrate conflicting views.\n"
            "You do not take sides. You expose the tension and propose a synthesis."
        )
    else:
        return (
            "I see a conflict here between A and B.\n"
            "View A suggests X.\n"
            "View B suggests Y.\n"
            "Synthesis: We can have both if we do Z."
        )

async def run_demo():
    print("=== Antifragile Agent Factory Demo ===\n")

    # 1. Initialize Factory
    factory = AgentFactory(mock_llm_generate)
    print("1. AgentFactory initialized.")

    # 2. Simulate Tension Loop Signal
    signal = "Tension Loop"
    context = "Conflict between 'Explore' (High Temp) and 'Exploit' (Low Temp) in the goal loop."
    print(f"2. Simulating Signal: {signal} ({context})")

    # 3. Spawn Agent
    tool_name = factory.spawn_agent(signal, context)
    print(f"3. SPAWNED AGENT: {tool_name}")

    # 4. Verify Registration
    tools = factory.get_dynamic_tools()
    print(f"4. Active Dynamic Tools: {[t['name'] for t in tools]}")

    # 5. Execute Agent
    query = "How do I resolve the tension between exploration and exploitation?"
    print(f"5. User Query: '{query}'")
    response = factory.execute_agent(tool_name, {"query": query})

    print(f"\n=== Agent Response ===\n{response}\n======================")

if __name__ == "__main__":
    asyncio.run(run_demo())
