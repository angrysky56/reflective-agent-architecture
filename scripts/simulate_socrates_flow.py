import asyncio
import logging

from src.compass.compass_framework import create_compass
from src.integration.sleep_cycle import SleepCycle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SocratesSimulation")

async def run_simulation():
    print("\n=== Initializing COMPASS with Authentic Socrates ===\n")
    compass = create_compass()

    # 1. Load Socrates
    socrates_id = "authentic_socrates"
    advisor = compass.advisor_registry.get_advisor(socrates_id)
    if not advisor:
        print("Error: Socrates not found. Please run install_socrates.py first.")
        return

    compass.integrated_intelligence.configure_advisor(advisor)
    print(f"Active Advisor: {compass.integrated_intelligence.current_advisor.name}")
    print(f"System Prompt: {compass.integrated_intelligence.current_advisor.system_prompt[:100]}...\n")

    # 2. Simulate User Interaction (The "Deep Chat")
    user_query = "What is the nature of true knowledge? Is it justified true belief?"
    print(f"User: {user_query}")

    # In a real app, this would go through the LLM.
    # Here we simulate the *cognitive steps* Socrates would take.

    print("\n--- Step 1: Deconstruction (The Elenchus) ---")
    # Socrates deconstructs the concept of "Knowledge" vs "Belief"
    deconstruct_result = compass.executive_controller.deconstruct_problem(user_query)
    print(f"Deconstruction Root: {deconstruct_result.get('root_node', 'Unknown')}")
    print(f"Fragments: {len(deconstruct_result.get('fragments', []))}")

    print("\n--- Step 2: Hypothesis (Finding Contradictions) ---")
    # Socrates looks for the gap between "Justification" and "Truth" (Gettier problems)
    # We simulate a hypothesis connecting these nodes
    hypo_result = compass.slap_pipeline.hypothesize(
        node_a_id="concept_knowledge",
        node_b_id="concept_justification",
        context="Epistemology"
    )
    print(f"Hypothesis: {hypo_result}")

    print("\n--- Step 3: Synthesis (Maieutics/Birthing Truth) ---")
    # Socrates helps the user realize that justification is fragile
    synthesis_goal = "Define knowledge beyond simple justification"
    # We create a dummy node ID for the synthesis to use in compression
    synthesis_node_id = "thought_synthesis_knowledge_v1"

    # Manually inject a synthesis node into the graph (mocking what synthesize() does)
    with compass.mcp_client.neo4j_driver.session() as session:
        session.run(
            """
            MERGE (n:ThoughtNode {id: $id})
            SET n.content = $content, n.cognitive_type = 'synthesis'
            """,
            id=synthesis_node_id,
            content="True knowledge requires an account (logos) that tethers the belief to reality, preventing it from running away like the statues of Daedalus."
        )
    print(f"Synthesized Insight: 'True knowledge requires an account (logos)...'")

    print("\n--- Step 4: Tool Creation (Crystallizing the Method) ---")
    # Socrates decides this "method of tethering beliefs" is a useful tool.
    # He calls it 'elenchus_probe'.

    tool_name = "elenchus_probe"
    description = "A tool to test the stability of a belief by demanding an account (logos) for it."

    print(f"Compressing insight into tool: {tool_name}")

    # We call the tool handler logic directly to ensure advisor linking happens
    # (In the real server, this is done in call_tool, but we can invoke the method and manually link for simulation)

    # 1. Create the tool in the workspace
    compass.mcp_client.compress_to_tool(
        node_ids=[synthesis_node_id],
        tool_name=tool_name,
        description=description
    )

    # 2. Link to Advisor (The Logic we just added to server.py)
    if tool_name not in advisor.tools:
        advisor.tools.append(tool_name)
        compass.advisor_registry.save_advisors()
        print(f"SUCCESS: Tool '{tool_name}' linked to advisor '{advisor.name}'")
    else:
        print(f"Tool '{tool_name}' already exists in advisor's toolkit.")

    print("\n--- Step 5: Sleep Cycle (Consolidation) ---")
    # Trigger a nap to consolidate this new memory
    sleep = SleepCycle(workspace=compass.mcp_client)
    nap_result = sleep.dream(epochs=1)
    print(f"Nap Result: {nap_result}")

    print("\n=== Simulation Complete ===")
    print("Socrates has analyzed knowledge, created the 'elenchus_probe' tool, and consolidated it.")

if __name__ == "__main__":
    asyncio.run(run_simulation())
