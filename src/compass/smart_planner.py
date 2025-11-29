"""
SMART Planner - Strategic Management & Resource Tracking

Implements SMART goal-oriented planning with measurable objectives,
feasibility assessment, timeline management, and progress monitoring.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .config import SMARTConfig
from .utils import COMPASSLogger, ObjectiveState


class SMARTPlanner:
    """
    SMART: Strategic Management & Resource Tracking

    Creates and manages SMART objectives (Specific, Measurable, Achievable,
    Relevant, Time-bound) for goal-oriented task execution.
    """

    def __init__(self, config, logger: Optional[COMPASSLogger] = None):
        """
        Initialize SMART planner.

        Args:
            config: SMARTConfig instance
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or COMPASSLogger("SMART")

        # Active objectives
        self.objectives: List[ObjectiveState] = []

        self.logger.info("SMART planner initialized")

    def create_objectives_from_task(self, task: str, context: Optional[Dict] = None) -> List[ObjectiveState]:
        """
        Define specific objectives for a task.

        Args:
            task: Task description
            context: Optional context

        Returns:
            List of ObjectiveState instances
        """
        self.logger.info("Creating SMART objectives from task")

        context = context or {}
        objectives = []

        # Analyze task to identify objectives
        if isinstance(task, dict):
            # Handle case where full semantic prompt is passed
            task = task.get("enriched_prompt", task.get("prompt", str(task)))

        task_lower = task.lower()

        # Default: create at least one objective
        primary_objective = self._create_primary_objective(task, context)
        objectives.append(primary_objective)

        # Create category-specific objectives
        if any(word in task_lower for word in ["learn", "improve", "optimize"]):
            learning_obj = self._create_learning_objective(task, context)
            objectives.append(learning_obj)

        if any(word in task_lower for word in ["decide", "choose", "select"]):
            decision_obj = self._create_decision_objective(task, context)
            objectives.append(decision_obj)

        if any(word in task_lower for word in ["fast", "quick", "efficient"]):
            performance_obj = self._create_performance_objective(task, context)
            objectives.append(performance_obj)

        # Assess feasibility and relevance for each
        for obj in objectives:
            obj.is_feasible = self._assess_feasibility(obj)
            obj.is_relevant = self._ensure_relevance(obj, task)

        # Filter out infeasible or irrelevant objectives
        objectives = [obj for obj in objectives if obj.is_feasible and obj.is_relevant]

        # Store objectives
        self.objectives.extend(objectives)

        self.logger.info(f"Created {len(objectives)} SMART objectives")
        return objectives

    def _create_primary_objective(self, task: str, context: Dict) -> ObjectiveState:
        """Create the primary objective from task."""
        # Extract task intent
        task_lower = task.lower()

        if "create" in task_lower or "build" in task_lower:
            name = "Creation"
            description = "Successfully create the requested artifact"
            metric = "Completeness score"
        elif "analyze" in task_lower:
            name = "Analysis"
            description = "Thoroughly analyze the subject"
            metric = "Analysis depth score"
        elif "optimize" in task_lower:
            name = "Optimization"
            description = "Optimize the target system or process"
            metric = "Improvement percentage"
        else:
            name = "Task Completion"
            description = "Complete the requested task"
            metric = "Success score"

        deadline = datetime.now() + timedelta(days=self.config.default_timeline_days)

        return ObjectiveState(name=name, description=description, metric=metric, target_value=1.0, current_value=0.0, deadline=deadline)

    def _create_learning_objective(self, task: str, context: Dict) -> ObjectiveState:
        """Create learning capability objective."""
        return ObjectiveState(name="Learning Enhancement", description="Improve learning capabilities through task execution", metric=self.config.metrics["learning-capabilities"], target_value=0.8, current_value=0.0, deadline=datetime.now() + timedelta(days=self.config.default_timeline_days))

    def _create_decision_objective(self, task: str, context: Dict) -> ObjectiveState:
        """Create decision-making objective."""
        return ObjectiveState(name="Decision Quality", description="Make high-quality decisions efficiently", metric=self.config.metrics["decision-making"], target_value=0.9, current_value=0.0, deadline=datetime.now() + timedelta(days=self.config.default_timeline_days))

    def _create_performance_objective(self, task: str, context: Dict) -> ObjectiveState:
        """Create performance optimization objective."""
        return ObjectiveState(name="Performance Optimization", description="Optimize for speed and efficiency", metric=self.config.metrics["performance-optimization"], target_value=0.85, current_value=0.0, deadline=datetime.now() + timedelta(days=self.config.default_timeline_days))

    def _assess_feasibility(self, objective: ObjectiveState) -> bool:
        """
        Analyze feasibility of an objective.

        Args:
            objective: Objective to assess

        Returns:
            True if feasible, False otherwise
        """
        # Simple heuristic: check if target value is reasonable
        if objective.target_value > 1.0 or objective.target_value < 0.0:
            self.logger.warning(f"Objective {objective.name} has invalid target value")
            return False

        # Check if deadline is in the future
        if objective.deadline < datetime.now():
            self.logger.warning(f"Objective {objective.name} has deadline in the past")
            return False

        return True

    def _ensure_relevance(self, objective: ObjectiveState, task: str) -> bool:
        """
        Align objective with task goals.

        Args:
            objective: Objective to check
            task: Original task

        Returns:
            True if relevant, False otherwise
        """
        # Simple check: objective name/description should relate to task
        task_words = set(task.lower().split())
        obj_words = set(objective.name.lower().split()) | set(objective.description.lower().split())

        # If there's overlap, it's relevant
        overlap = task_words & obj_words

        return len(overlap) > 0 or objective.name == "Task Completion"

    def monitor_progress(self, objectives: Optional[List[ObjectiveState]] = None) -> Dict[str, any]:
        """
        Monitor progress on objectives.

        Args:
            objectives: Objectives to monitor (uses all if None)

        Returns:
            Progress report dictionary
        """
        objectives = objectives or self.objectives

        report = {"total_objectives": len(objectives), "on_track": 0, "off_track": 0, "completed": 0, "details": []}

        for obj in objectives:
            status = self._check_objective_status(obj)

            if status == "completed":
                report["completed"] += 1
            elif status == "on_track":
                report["on_track"] += 1
            else:
                report["off_track"] += 1

            report["details"].append({"name": obj.name, "status": status, "progress": obj.progress, "is_on_track": obj.is_on_track})

        self.logger.info(f"Progress: {report['completed']} completed, {report['on_track']} on track, {report['off_track']} off track")

        return report

    def _check_objective_status(self, objective: ObjectiveState) -> str:
        """Check objective status."""
        if objective.progress >= 100.0:
            return "completed"
        elif objective.is_on_track:
            return "on_track"
        else:
            return "off_track"

    def adjust_objective(self, objective: ObjectiveState, adjustments: Dict) -> ObjectiveState:
        """
        Make adjustments to an objective.

        Args:
            objective: Objective to adjust
            adjustments: Dictionary of adjustments

        Returns:
            Adjusted objective
        """
        self.logger.info(f"Adjusting objective: {objective.name}")

        if "target_value" in adjustments:
            objective.target_value = adjustments["target_value"]

        if "deadline" in adjustments:
            objective.deadline = adjustments["deadline"]

        if "description" in adjustments:
            objective.description = adjustments["description"]

        # Re-assess feasibility
        objective.is_feasible = self._assess_feasibility(objective)

        return objective

    def get_objectives_summary(self) -> str:
        """
        Get a text summary of all objectives.

        Returns:
            Formatted summary string
        """
        if not self.objectives:
            return "No objectives defined."

        summary = "SMART Objectives Summary:\n\n"

        for i, obj in enumerate(self.objectives, 1):
            summary += f"{i}. {obj.name}\n"
            summary += f"   Description: {obj.description}\n"
            summary += f"   Metric: {obj.metric}\n"
            summary += f"   Progress: {obj.progress:.1f}%\n"
            summary += f"   Target: {obj.target_value}, Current: {obj.current_value:.2f}\n"
            summary += f"   Deadline: {obj.deadline.strftime('%Y-%m-%d')}\n"
            summary += f"   Status: {'✓ On Track' if obj.is_on_track else '✗ Off Track'}\n\n"

        return summary

    def reset(self):
        """Reset planner state."""
        self.objectives.clear()
        self.logger.debug("SMART planner reset")
