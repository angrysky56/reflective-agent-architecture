"""
Executive Controller - Meta-Cognitive Control System

Implements the Executive Controller module of the CGRA architecture.
Coordinates oMCD (resource allocation) and Self-Discover (reflection)
to manage goals, strategies, and continuous evaluation.
"""

from typing import Any, Dict, List, Optional

from .advisors import AdvisorProfile, AdvisorRegistry
from .config import ExecutiveControllerConfig, SelfDiscoverConfig, oMCDConfig
from .omcd_controller import oMCDController
from .self_discover_engine import SelfDiscoverEngine
from .utils import COMPASSLogger, Trajectory


class ExecutiveController:
    """
    Unified meta-cognitive controller that coordinates reasoning, resources, and reflection.
    Acts as the central "brain" of the agent.
    """

    def __init__(self, config: ExecutiveControllerConfig, omcd_config: oMCDConfig, self_discover_config: SelfDiscoverConfig, advisor_registry: AdvisorRegistry, logger: Optional[COMPASSLogger] = None):
        """
        Initialize Executive Controller.

        Args:
            config: Executive Controller configuration
            omcd_config: Configuration for oMCD
            self_discover_config: Configuration for Self-Discover
            advisor_registry: Shared Advisor Registry
            logger: Optional logger
        """
        self.config = config
        self.omcd_config = omcd_config
        self.self_discover_config = self_discover_config
        self.logger = logger or COMPASSLogger("ExecutiveController")

        # Sub-controllers
        self.omcd = oMCDController(omcd_config, logger=self.logger)
        self.self_discover = SelfDiscoverEngine(self_discover_config, logger=self.logger)

        # State tracking
        self.current_strategy: List[int] = []
        self.context: Dict[str, Any] = {}
        self.iteration_count: int = 0

        # Gödel Check State
        self.godel_loop_count: int = 0
        self.godel_threshold: int = 3  # Cycles of disagreement before override

        # Shared Advisor Registry
        self.advisor_registry = advisor_registry

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

    def coordinate_iteration(self, task: str, current_state: Dict, context: Optional[Dict] = None) -> Dict[str, Any]:
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
        if not self.current_strategy or (self.iteration_count % self.config.strategy_update_interval == 0):
            reflections = self.self_discover.get_memory()
            self.current_strategy = self.self_discover.select_reasoning_modules(task, reflections, context=self.context)

        # 4. Resource Allocation (oMCD)
        # Calculate importance based on goal priority and solvability
        importance = max(0.01, (current_goal.priority if current_goal else 0.5) * solvability.get("solvability_score", 0.5))

        allocation = self.omcd.determine_resource_allocation(current_state, importance=importance, available_resources=self.omcd.config.max_resources - self.omcd.total_resources_allocated)

        # 5. Gödel Check (Sanity Loop)
        # Check for infinite regress: Director wants to proceed (allocation > 0) but Reality/Swarm says NO (low solvability)
        godel_override = self._check_godel_loop(allocation, solvability)

        if godel_override:
            self.logger.critical(f"Gödel Check TRIGGERED: Infinite Regress detected for {self.godel_threshold} cycles. Initiating Human Override.")
            return {
                "goal": current_goal,
                "strategy": self.current_strategy,
                "resources": allocation,
                "solvability": solvability,
                "should_stop": True,
                "godel_override": True,
                "reason": "Gödel Incompleteness: System cannot verify its own consistency."
            }

        return {"goal": current_goal, "strategy": self.current_strategy, "resources": allocation, "solvability": solvability, "should_stop": self.omcd.should_stop(current_score=current_state.get("score", 0.0), iteration=self.iteration_count, allocation=allocation, target_score=self.self_discover.config.pass_threshold)}

    def _check_godel_loop(self, allocation: Dict, solvability: Dict) -> bool:
        """
        Perform a 'Gödel Check' to detect if the system is trapped in a self-verification loop.

        Logic:
        - If the Director allocates resources (trying to solve)
        - BUT the Solvability score is low (Reality/Swarm indicates impossibility/high entropy)
        - AND this persists for X cycles
        -> Then we are likely in an undecidable state (Halting Problem).

        Args:
            allocation: Resource allocation decision
            solvability: Task solvability assessment

        Returns:
            True if Human Override is required.
        """
        # Thresholds
        RESOURCE_THRESHOLD = 0.1  # Director is trying
        SOLVABILITY_THRESHOLD = 0.4  # Reality says "Hard/Impossible"

        is_trying = allocation.get("amount", 0.0) > RESOURCE_THRESHOLD
        is_stuck = solvability.get("solvability_score", 0.5) < SOLVABILITY_THRESHOLD

        if is_trying and is_stuck:
            self.godel_loop_count += 1
            self.logger.warning(f"Gödel Check Warning: Cycle {self.godel_loop_count}/{self.godel_threshold}. Director pushing against high entropy.")
        else:
            # Reset if we stabilize or stop trying
            if self.godel_loop_count > 0:
                self.logger.info("Gödel Check Reset: System stabilized.")
            self.godel_loop_count = 0

        return self.godel_loop_count >= self.godel_threshold

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

    def update_goal_status(self, goal_id: str, progress: float, status: str = "active"):
        """Update status of a goal."""
        self.omcd.adjust_goal(goal_id, {"progress": progress, "status": status})

    def reset(self):
        """Reset controller state."""
        self.omcd = oMCDController(self.omcd.config, logger=self.logger)  # Re-init to clear state
        self.self_discover.reset()
        self.current_strategy = []
        self.iteration_count = 0
        self.logger.debug("Executive Controller reset")
