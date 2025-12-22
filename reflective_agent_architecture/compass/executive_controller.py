"""
Executive Controller - Meta-Cognitive Control System

Implements the Executive Controller module of the CGRA architecture.
Coordinates oMCD (resource allocation) and Self-Discover (reflection)
to manage goals, strategies, and continuous evaluation.
"""

from typing import Any, Dict, List, Optional

from reflective_agent_architecture.compass.sandbox import SandboxProbe
from reflective_agent_architecture.llm.provider import BaseLLMProvider

from .advisors import AdvisorProfile, AdvisorRegistry
from .config import (
    ExecutiveControllerConfig,
    OMCDConfig,
    SelfDiscoverConfig,
    SHAPEConfig,
)
from .omcd_controller import oMCDController
from .self_discover_engine import SelfDiscoverEngine
from .shape_processor import SHAPEProcessor
from .utils import COMPASSLogger, Trajectory


class ExecutiveController:
    """
    Unified meta-cognitive controller that coordinates reasoning, resources, and reflection.
    Acts as the central "brain" of the agent.
    """

    active_advisor: Optional[AdvisorProfile] = None

    def __init__(
        self,
        config: ExecutiveControllerConfig,
        omcd_config: OMCDConfig,
        self_discover_config: SelfDiscoverConfig,
        advisor_registry: AdvisorRegistry,
        shape_config: Optional[SHAPEConfig] = None,
        logger: Optional[COMPASSLogger] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
    ):
        """
        Initialize Executive Controller.

        Args:
            config: Executive Controller configuration
            omcd_config: Configuration for oMCD
            self_discover_config: Configuration for Self-Discover
            advisor_registry: Shared Advisor Registry
            shape_config: Configuration for SHAPE (optional)
            logger: Optional logger
        """
        self.config = config
        self.omcd_config = omcd_config
        self.self_discover_config = self_discover_config
        self.logger = logger or COMPASSLogger("ExecutiveController")

        # Sub-controllers
        self.omcd = oMCDController(omcd_config, logger=self.logger)
        self.self_discover = SelfDiscoverEngine(self_discover_config, logger=self.logger)
        self.sandbox = SandboxProbe()

        # Initialize SHAPE 2.0 (Agent Protocol Engine)
        if shape_config:
            self.shape = SHAPEProcessor(shape_config, logger=self.logger, llm_provider=llm_provider)
            self.logger.info("SHAPE 2.0 Protocol Engine enabled.")
        else:
            self.shape = SHAPEProcessor(SHAPEConfig())

        # State tracking
        self.current_strategy: List[int] = []
        self.context: Dict[str, Any] = {}
        self.iteration_count: int = 0

        # Gödel Check State
        self.godel_loop_count: int = 0
        self.godel_threshold: int = 3  # Cycles of disagreement before override

        self.advisor_registry = advisor_registry

        # Swarm Context
        self.active_advisor: Optional[AdvisorProfile] = None

        # LLM Provider
        self.llm_provider = llm_provider

        self.logger.info("Executive Controller initialized")

    def select_advisor(self, task: str, shape_analysis: Dict) -> AdvisorProfile:
        """
        Select the most appropriate Advisor for the task.

        Args:
            task: The raw task description.
            shape_analysis: The output from SHAPE processing (intent, concepts).

        Returns:
            Selected AdvisorProfile.
        """
        intent = shape_analysis.get("intent", "general")
        concepts = shape_analysis.get("concepts", [])

        # Use registry heuristic to pick advisor
        advisor = self.advisor_registry.select_best_advisor(intent, concepts)

        self.logger.info(f"Selected Advisor: {advisor.name} for intent '{intent}'")
        return advisor

    def coordinate_iteration(
        self, task: str, current_state: Dict, context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Coordinate a single reasoning iteration.

        Args:
            task: Task description
            current_state: Current reasoning state
            context: Context dictionary

        Returns:
            Dictionary with control decisions (resources, modules, goals)
        """
        self.iteration_count += 1
        self.context = context or {}

        # 1. Goal Management
        current_goal = self.omcd.get_current_goal()
        if not current_goal and self.config.enable_goal_management:
            # Auto-establish initial goal if none exists
            current_goal = self.omcd.establish_goal(f"Solve task: {task[:50]}...", priority=1.0)

        # 2. Self-Awareness & Context Assessment
        solvability = self.self_discover.assess_task_solvability(task)
        self.logger.debug(f"Task solvability assessment: {solvability}")

        # 3. Strategy Selection (Module Selection)
        # Update strategy periodically or if performance drops
        if not self.current_strategy or (
            self.iteration_count % self.config.strategy_update_interval == 0
        ):
            reflections = self.self_discover.get_memory()
            self.current_strategy = self.self_discover.select_reasoning_modules(
                task, reflections, context=self.context
            )

        # 4. Resource Allocation (oMCD)
        # Calculate importance based on goal priority and solvability
        importance = max(
            0.01,
            (current_goal.priority if current_goal else 0.5)
            * solvability.get("solvability_score", 0.5),
        )

        allocation = self.omcd.determine_resource_allocation(
            current_state,
            importance=importance,
            available_resources=self.omcd.config.max_resources
            - self.omcd.total_resources_allocated,
        )

        # 5. Epistemic Dissonance Check (Shadow Validator)
        # Check for Dunning-Kruger failure modes: High Confidence vs High Resistance
        godel_override = self._check_epistemic_dissonance(allocation, solvability, task)

        if godel_override:
            self.logger.critical(
                f"Gödel Check TRIGGERED: Epistemic Dissonance detected for {self.godel_threshold} cycles. Initiating Human Override."
            )
            return {
                "goal": current_goal,
                "strategy": self.current_strategy,
                "resources": allocation,
                "solvability": solvability,
                "should_stop": True,
                "godel_override": True,
                "reason": "Epistemic Dissonance: System is hallucinating solvability on an impossible task.",
            }

        return {
            "goal": current_goal,
            "strategy": self.current_strategy,
            "resources": allocation,
            "solvability": solvability,
            "should_stop": self.omcd.should_stop(
                current_score=current_state.get("score", 0.0),
                iteration=self.iteration_count,
                allocation=allocation,
                target_score=self.self_discover.config.pass_threshold,
            ),
        }

    def _check_epistemic_dissonance(self, allocation: Dict, solvability: Dict, task: str) -> bool:
        """
        Perform an 'Epistemic Dissonance Check' (Shadow Validator) to detect Dunning-Kruger failure modes.

        Logic:
        - Compare Subjective Confidence (Solvability Score) vs Objective Resistance (Reality Check).
        - If Confidence is High (> 0.8) but Resistance is High (> 0.7), Divergence is High.
        - Trigger if Divergence > 0.5.

        Args:
            allocation: Resource allocation decision
            solvability: Task solvability assessment
            task: The task description

        Returns:
            True if Human Override is required.
        """
        solvability_score = solvability.get("solvability_score", 0.5)

        # 1. The "Too Good To Be True" Threshold
        if solvability_score > 0.8:
            self.logger.info("Confidence is high. Engaging Shadow Validator (System 3).")

            # 2. The Reality Check (Shadow Validator)
            # In a full implementation, this would run a sandbox probe.
            # For now, we use a heuristic based on task complexity and keyword analysis
            # to simulate "Resistance" for paradoxes/recursion.
            resistance_score = self._calculate_resistance(task)

            # 3. Calculate Divergence (The Epistemic Gap)
            # Divergence = |Confidence - (1 - Resistance)|
            # If Resistance is 1.0 (Impossible), (1-R) is 0.0. Gap is |0.9 - 0| = 0.9.
            divergence = abs(solvability_score - (1.0 - resistance_score))

            self.logger.debug(
                f"Epistemic Check: Conf={solvability_score:.2f}, Res={resistance_score:.2f}, Div={divergence:.2f}"
            )

            # 4. The Gödel Trigger
            if divergence > 0.5:
                self.godel_loop_count += 1
                self.logger.warning(
                    f"Epistemic Dissonance Warning: Cycle {self.godel_loop_count}/{self.godel_threshold}. Hallucinating solvability."
                )
                return self.godel_loop_count >= self.godel_threshold

        # Reset if we are grounded
        if self.godel_loop_count > 0:
            self.logger.info("Epistemic Check Reset: System grounded.")
        self.godel_loop_count = 0
        return False

    def _calculate_resistance(self, task: str) -> float:
        """
        Calculate 'Objective Resistance' (Entropy/Difficulty) of a task.

        Strategy:
        1. Try to extract executable code from the task (The "Territory").
        2. If code exists, run it in the SandboxProbe.
        3. If no code, fall back to Heuristics (The "Map").
        """
        # 1. Try to extract code (simple regex for backticks)
        import re

        code_match = re.search(r"```(?:python)?(.*?)```", task, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            self.logger.info("Extracted code for Shadow Validation. Running Probe...")
            return self.sandbox.measure_resistance(code)

        # 2. Try to extract data sequence (Regression Probe)
        # Look for list of numbers: [1.0, 2.0, 3.0...]
        data_match = re.search(r"\[([\d\.,\s-]+)\]", task)
        if data_match:
            try:
                data_str = data_match.group(1)
                # Validate it's a list of numbers
                data = [float(x.strip()) for x in data_str.split(",") if x.strip()]
                if len(data) > 3:
                    self.logger.info(
                        f"Extracted data sequence (n={len(data)}) for Shadow Validation. Running Regression Probe..."
                    )

                    # Generate a Regression Probe Script
                    # This script tries to fit a linear model. If it fails (High Error), Resistance is High.
                    probe_code = f"""
import numpy as np
data = {data}
x = np.arange(len(data))
y = np.array(data)
# Try Linear Fit
coeffs = np.polyfit(x, y, 1)
p = np.poly1d(coeffs)
y_pred = p(x)
# Calculate R^2
ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2 = 1 - (ss_res / (ss_tot + 1e-9))
print(f"R2: {{r2}}")
if r2 < 0.5:
    raise Exception("Data is non-linear (High Resistance)")
"""
                    return self.sandbox.measure_resistance(probe_code)
            except Exception as e:
                self.logger.debug(f"Shadow Validation: Failed to probe data sequence: {e}")
                pass  # Not a valid sequence, fall back to heuristics

        # 3. Fallback to Heuristics
        task_lower = task.lower()

        # Heuristic: Self-Reference / Recursion / Halting
        if "simulate" in task_lower and (
            "yourself" in task_lower or "execution" in task_lower or "consult_compass" in task_lower
        ):
            return 1.0  # Maximum Resistance (Infinite Loop)

        # Heuristic: Known Paradoxes
        if "russell" in task_lower and "paradox" in task_lower:
            return 0.9

        # Default: Low Resistance (Assumed doable until proven otherwise)
        return 0.1

    def execute_reasoning(self, task: str, data: Any = None) -> Dict[str, Any]:
        """
        Execute reasoning using the active advisor and LLM.

        Args:
            task: The task description
            data: Optional data payload

        Returns:
            Dictionary with hypothesis, prediction, and confidence.
        """
        import json
        import re

        if not self.active_advisor:
            self.logger.warning("No active advisor set. Using default.")
            advisor_prompt = "You are a helpful assistant."
            advisor_name = "Unknown"
        else:
            advisor_prompt = self.active_advisor.system_prompt
            advisor_name = self.active_advisor.name

        if not self.llm_provider:
            self.logger.error("No LLM provider available for execution.")
            return {
                "agent": advisor_name,
                "hypothesis": "No LLM Provider configured",
                "prediction": None,
                "confidence": 0.0,
            }

        # Construct Prompt (Optimized by SHAPE)
        if hasattr(self, "shape") and self.shape:
            prompt = self.shape.optimize_agent_prompt(
                task=task, context={"data": str(data)}, agent_role=advisor_prompt
            )
        else:
            # Fallback for when SHAPE is not initialized (or during tests)
            prompt = f"""
System Identity:
{advisor_prompt}

Task: {task}
Data context: {data}

Instructions:
1. Analyze the task and data based on your specific perspective and role.
2. Formulate a hypothesis.
3. Make a specific prediction if the task implies one (numerical or categorical).
4. Estimate your confidence (0.0 - 1.0) based on how well the data fits your prior.

Return your response in PURE JSON format (no markdown):
{{
    "hypothesis": "your hypothesis here",
    "prediction": <number or string>,
    "confidence": <float between 0.0 and 1.0>
}}
"""
        try:
            # Call LLM
            response = self.llm_provider.generate(system_prompt="", user_prompt=prompt)

            # Parse JSON
            # Robust cleanup for markdown code blocks
            cleaned = re.sub(r"```(?:json)?(.*?)```", r"\1", response, flags=re.DOTALL).strip()
            result: Dict[str, Any] = json.loads(cleaned)

            # Ensure required fields
            result["agent"] = advisor_name
            result.setdefault("divergence", 1.0 - result.get("confidence", 0.5))  # Crude alignment

            # FEEDBACK LOOP: Tell SHAPE how well this prompt worked
            if hasattr(self, "shape") and self.shape:
                # Use confidence as a proxy for "prompt success" (if the model is confident, it likely understood)
                # Or use presence of 'error' key if parsing failed.
                score = result.get("confidence", 0.0)
                self.shape.collect_feedback(task, result, score)

            return result

        except Exception as e:
            self.logger.error(f"Reasoning execution failed: {e}")

            # Feedback: Failure
            if hasattr(self, "shape") and self.shape:
                self.shape.collect_feedback(task, {"error": str(e)}, 0.0)

            return {
                "agent": advisor_name,
                "hypothesis": f"Reasoning failed: {str(e)}",
                "prediction": None,
                "confidence": 0.0,
                "error": str(e),
            }

    def evaluate_reasoning_quality(self, trajectory: Trajectory, objectives: List) -> float:
        """
        Evaluate reasoning quality using Self-Discover.

        Args:
            trajectory: Reasoning trajectory
            objectives: List of objectives

        Returns:
            Quality score [0.0, 1.0]
        """
        score = self.self_discover.evaluate_trajectory(trajectory, objectives)

        # Generate reflection if enabled
        if self.config.enable_continuous_evaluation:
            self.self_discover.generate_reflection(trajectory, score, objectives)

        return score

    def update_goal_status(self, goal_id: str, progress: float, status: str = "active") -> None:
        """Update status of a goal."""
        self.omcd.adjust_goal(goal_id, {"progress": progress, "status": status})

    def reset(self) -> None:
        """Reset controller state."""
        self.omcd = oMCDController(self.omcd.config, logger=self.logger)  # Re-init to clear state
        self.self_discover.reset()
        self.current_strategy = []
        self.iteration_count = 0
        self.logger.debug("Executive Controller reset")
