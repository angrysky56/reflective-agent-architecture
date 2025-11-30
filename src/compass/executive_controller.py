"""
Executive Controller - Meta-Cognitive Control System

Implements the Executive Controller module of the CGRA architecture.
Coordinates oMCD (resource allocation) and Self-Discover (reflection)
to manage goals, strategies, and continuous evaluation.
"""

from typing import Any, Dict, List, Optional

from .config import ExecutiveControllerConfig, SelfDiscoverConfig, oMCDConfig
from .omcd_controller import oMCDController
from .self_discover_engine import SelfDiscoverEngine
from .utils import COMPASSLogger, Trajectory


class ExecutiveController:
    """
    Unified meta-cognitive controller that coordinates reasoning, resources, and reflection.
    Acts as the central "brain" of the agent.
    """

    def __init__(self, config: ExecutiveControllerConfig, omcd_config: oMCDConfig, self_discover_config: SelfDiscoverConfig, logger: Optional[COMPASSLogger] = None):
        """
        Initialize Executive Controller.

        Args:
            config: Executive Controller configuration
            omcd_config: Configuration for oMCD
            self_discover_config: Configuration for Self-Discover
            logger: Optional logger
        """
        self.config = config
        self.logger = logger or COMPASSLogger("ExecutiveController")

        # Sub-controllers
        self.omcd = oMCDController(omcd_config, logger=self.logger)
        self.self_discover = SelfDiscoverEngine(self_discover_config, logger=self.logger)

        # State tracking
        self.current_strategy: List[int] = []
        self.context: Dict[str, Any] = {}
        self.iteration_count: int = 0

        self.logger.info("Executive Controller initialized")

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

        return {"goal": current_goal, "strategy": self.current_strategy, "resources": allocation, "solvability": solvability, "should_stop": self.omcd.should_stop(current_score=current_state.get("score", 0.0), iteration=self.iteration_count, allocation=allocation, target_score=self.self_discover.config.pass_threshold)}

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
