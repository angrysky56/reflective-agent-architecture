import json
import os

from src.compass.advisors.registry import AdvisorProfile, AdvisorRegistry


def verify_advisor_learning():
    print("Verifying Advisor Learning (Tool Association)...")

    test_storage = "tests/test_advisors_learning.json"
    if os.path.exists(test_storage):
        os.remove(test_storage)

    # 1. Initialize Registry
    registry = AdvisorRegistry(storage_path=test_storage)

    # 2. Create Advisor
    advisor = AdvisorProfile(
        id="learner",
        name="Learning Advisor",
        role="Student",
        description="Learns new tools.",
        system_prompt="I learn.",
        tools=["existing_tool"]
    )
    registry.register_advisor(advisor)
    print("Advisor created with initial tools.")

    # 3. Simulate Learning (Adding a tool)
    print("Simulating tool learning...")
    new_tool = "newly_learned_tool"
    advisor.tools.append(new_tool)
    registry.save_advisors()
    print(f"Added '{new_tool}' and saved.")

    # 4. Reload and Verify
    print("Reloading registry...")
    registry2 = AdvisorRegistry(storage_path=test_storage)
    loaded_advisor = registry2.get_advisor("learner")

    assert loaded_advisor is not None
    assert "existing_tool" in loaded_advisor.tools
    assert "newly_learned_tool" in loaded_advisor.tools
    print("Verification Successful: Advisor remembered the new tool!")

    # 5. Verify List Advisors Output Format (Simulation)
    print("\nSimulating list_advisors output:")
    result_lines = ["Available Advisors:"]
    for adv in registry2.advisors.values():
        result_lines.append(f"- {adv.name} ({adv.id}): {adv.role}")
        result_lines.append(f"  Description: {adv.description}")
        result_lines.append(f"  Tools: {', '.join(adv.tools)}")
        result_lines.append("")
    output = "\n".join(result_lines)
    print(output)
    assert "newly_learned_tool" in output

    # Cleanup
    if os.path.exists(test_storage):
        os.remove(test_storage)

if __name__ == "__main__":
    verify_advisor_learning()
