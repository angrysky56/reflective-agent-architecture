import asyncio
import json
import os

from src.compass.advisors.registry import AdvisorProfile, AdvisorRegistry


def verify_persistence():
    print("Verifying Advisor Persistence...")

    # Use a temporary test file
    test_storage = "tests/test_advisors.json"
    if os.path.exists(test_storage):
        os.remove(test_storage)

    # 1. Initialize Registry and Create Advisor
    print("1. Initializing Registry and Creating Advisor...")
    registry1 = AdvisorRegistry(storage_path=test_storage)

    test_advisor = AdvisorProfile(
        id="persistence_test",
        name="Persistence Test",
        role="Tester",
        description="A temporary advisor for testing.",
        system_prompt="You are a test."
    )
    registry1.register_advisor(test_advisor)

    assert "persistence_test" in registry1.advisors
    assert os.path.exists(test_storage)
    print("Advisor created and file saved.")

    # 2. Re-initialize Registry (Simulate Restart)
    print("\n2. Simulating Restart (Re-initializing Registry)...")
    registry2 = AdvisorRegistry(storage_path=test_storage)

    assert "persistence_test" in registry2.advisors
    loaded_advisor = registry2.get_advisor("persistence_test")
    assert loaded_advisor.name == "Persistence Test"
    print("Advisor successfully loaded from disk.")

    # 3. Delete Advisor
    print("\n3. Deleting Advisor...")
    registry2.remove_advisor("persistence_test")
    assert "persistence_test" not in registry2.advisors

    # 4. Re-initialize again to verify deletion persisted
    print("\n4. Verifying Deletion Persistence...")
    registry3 = AdvisorRegistry(storage_path=test_storage)
    assert "persistence_test" not in registry3.advisors
    print("Deletion successfully persisted.")

    # Cleanup
    if os.path.exists(test_storage):
        os.remove(test_storage)

    print("\nAdvisor Persistence Verification Passed!")

if __name__ == "__main__":
    verify_persistence()
