import asyncio

from src.compass.compass_framework import create_compass


async def test_dynamic_advisor_creation():
    print("Testing Dynamic Advisor Creation...")

    # 1. Initialize COMPASS
    compass = create_compass()
    print("COMPASS initialized.")

    # 2. Create "Socrates" Advisor via IntegratedIntelligence
    print("\nCreating 'Socrates' Advisor...")
    result = compass.integrated_intelligence.create_advisor(
        id="socrates",
        name="Socrates",
        role="Philosopher",
        description="Uses the Socratic method to explore concepts.",
        system_prompt="You are Socrates. Answer questions with questions to guide the user to the truth.",
        tools=["deconstruct", "hypothesize"]
    )
    print(f"Creation Result: {result}")

    # 3. Verify Registration
    advisor = compass.advisor_registry.get_advisor("socrates")
    assert advisor is not None
    assert advisor.name == "Socrates"
    print("Socrates successfully registered.")

    # 4. Test Selection (Mock SHAPE result)
    # Note: The current heuristic in AdvisorRegistry is simple (keywords).
    # Since Socrates isn't hardcoded in the heuristic, we might need to update the heuristic
    # OR just verify that we *can* select him manually or if we update the heuristic.
    # For this test, let's just verify he exists and can be retrieved.
    # To test automatic selection, we'd need to update select_best_advisor to be smarter or include "philosophy" keywords.

    # Let's manually select him to prove he's available for selection
    compass.integrated_intelligence.configure_advisor(advisor)
    assert compass.integrated_intelligence.current_advisor.id == "socrates"
    print("IntegratedIntelligence successfully configured as Socrates.")

    # 5. Test Deletion
    print("\nTesting Advisor Deletion...")
    delete_result = compass.integrated_intelligence.delete_advisor("socrates")
    print(f"Deletion Result: {delete_result}")

    # Verify removal
    deleted_advisor = compass.advisor_registry.get_advisor("socrates")
    assert deleted_advisor is None
    print("Socrates successfully removed from registry.")

    print("\nDynamic Advisor Lifecycle (Create/Delete) Verification Passed!")

if __name__ == "__main__":
    asyncio.run(test_dynamic_advisor_creation())
