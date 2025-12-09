import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from compass.advisors import AdvisorRegistry
from compass.config import ExecutiveControllerConfig, OMCDConfig, SelfDiscoverConfig, SHAPEConfig
from compass.executive_controller import ExecutiveController
from compass.utils import COMPASSLogger


class MockLLM:
    def generate(self, system_prompt, user_prompt, **kwargs):
        return """```json
{"hypothesis": "Test hypothesis", "prediction": 0.8, "confidence": 0.9}
```"""


def test_shape_integration():
    logger = COMPASSLogger("Test")

    # 1. Initialize configs
    config = ExecutiveControllerConfig()
    omcd_config = OMCDConfig()
    sd_config = SelfDiscoverConfig()
    shape_config = SHAPEConfig()  # With default agent_optimization_mode

    advisor_registry = AdvisorRegistry()
    mock_llm = MockLLM()

    # 2. Initialize ExecutiveController WITH SHAPE
    agent = ExecutiveController(
        config,
        omcd_config,
        sd_config,
        advisor_registry,
        shape_config=shape_config,
        logger=logger,
        llm_provider=mock_llm,
    )

    # Manually set advisor
    agent.active_advisor = advisor_registry.get_advisor("linearist")

    print("Agent initialized. Checking SHAPE...")
    if hasattr(agent, "shape") and agent.shape:
        print("PASS: SHAPE initialized in ExecutiveController.")
    else:
        print("FAIL: SHAPE not initialized.")

    # 3. specific test for optimization
    task = "opt prompt for speed"
    print(f"Testing optimization with task: '{task}'")

    # We can't easily spy on the internal call without deeper mocking,
    # but we can check if execution runs without error and returns result.
    result = agent.execute_reasoning(task, data=[1, 2, 3])

    print("Execution Result:", result)

    if result["confidence"] == 0.9:
        print("PASS: Execution successful.")
    else:
        print("FAIL: Execution returned unexpected result.")

    # 4. Check feedback loop
    # SHAPE should have received feedback
    if len(agent.shape.expansion_feedback) > 0:
        print(f"PASS: SHAPE received feedback: {agent.shape.expansion_feedback}")
    else:
        print("FAIL: SHAPE did not receive feedback.")


if __name__ == "__main__":
    test_shape_integration()
