import asyncio

from src.compass.advisors.registry import AdvisorRegistry
from src.compass.compass_framework import create_compass


async def test_advisor_system():
    print("Testing Native Advisor System...")

    # 1. Initialize COMPASS
    compass = create_compass()
    print("COMPASS initialized.")

    # 2. Mock SHAPE result (Intent: Research)
    task = "Find the latest papers on causal inference."
    shape_result = {"intent": "research", "concepts": ["causal inference", "papers"]}

    # 3. Test Advisor Selection (Manual Check)
    print("\nTesting Advisor Selection Logic:")
    registry = AdvisorRegistry()
    advisor = registry.select_best_advisor(shape_result["intent"], shape_result["concepts"])
    print(f"Intent: '{shape_result['intent']}' -> Selected Advisor: {advisor.name} ({advisor.role})")
    assert advisor.id == "researcher"

    # 4. Mock SHAPE result (Intent: Coding)
    task_code = "Fix the bug in the server."
    shape_result_code = {"intent": "fix bug", "concepts": ["server", "bug"]}
    advisor_code = registry.select_best_advisor(shape_result_code["intent"], shape_result_code["concepts"])
    print(f"Intent: '{shape_result_code['intent']}' -> Selected Advisor: {advisor_code.name} ({advisor_code.role})")
    assert advisor_code.id == "coder"

    print("\nAdvisor System Verification Passed!")

if __name__ == "__main__":
    asyncio.run(test_advisor_system())
