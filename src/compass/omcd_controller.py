"""
oMCD Controller - Online Metacognitive Control of Decisions

Implements the oMCD model for optimal resource allocation based on
confidence-cost tradeoffs and metacognitive control.
"""

import uuid
from typing import Dict, List, Optional

import numpy as np

from .config import OMCDConfig
from .utils import (
    COMPASSLogger,
    Goal,
    calculate_benefit,
    calculate_cost,
    confidence_from_delta_mu,
    update_precision,
    validate_non_negative,
    validate_positive,
)


class oMCDController:
    """
    oMCD: Online Metacognitive Control of Decisions

    Optimizes cognitive resource allocation by balancing confidence benefits
    against effort costs using a Markov Decision Process framework.
    """

    def __init__(self, config: OMCDConfig, logger: Optional[COMPASSLogger] = None):
        """
        Initialize oMCD controller.

        Args:
            config: OMCDConfig instance
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or COMPASSLogger("oMCD")

        # State tracking
        self.current_confidence = 0.5
        self.total_resources_allocated = 0.0
        self.decision_history: List[Dict[str, float]] = []

        # Goal Management (CGRA)
        self.goal_stack: List[Goal] = []

        self.logger.info("oMCD controller initialized")

    def determine_resource_allocation(
        self, current_state: Dict, importance: float, available_resources: float
    ) -> Dict[str, float]:
        """
        Determine optimal cognitive resource allocation.

        Args:
            current_state: Current decision state
            importance: Importance weight (R parameter)
            available_resources: Resources available

        Returns:
            Resource allocation dictionary with amount, expected benefit, cost
        """
        validate_non_negative(available_resources, "available_resources")
        validate_positive(importance, "importance")

        # Extract value representations from state
        initial_precision = current_state.get("precision", 1.0)
        value_difference = current_state.get("value_difference", 0.0)
        variance = current_state.get("variance", 1.0)

        # Calculate optimal resource allocation
        optimal_z = self._calculate_optimal_allocation(
            initial_precision, value_difference, variance, importance, available_resources
        )

        # Calculate confidence at optimal allocation
        updated_precision = update_precision(initial_precision, optimal_z, self.config.beta)

        confidence = confidence_from_delta_mu(
            value_difference,
            1.0 / updated_precision,  # Convert precision to variance
            self.config.lambda_conf,
        )

        # Calculate benefit and cost
        benefit = calculate_benefit(optimal_z, importance, confidence)
        cost = calculate_cost(optimal_z, self.config.alpha, self.config.nu)

        # Update state
        self.current_confidence = confidence
        self.total_resources_allocated += optimal_z

        allocation = {
            "amount": optimal_z,
            "confidence": confidence,
            "benefit": benefit,
            "cost": cost,
            "net_benefit": benefit - cost,
            "updated_precision": updated_precision,
        }

        self.decision_history.append(allocation)

        self.logger.debug(
            f"Resource allocation: z={optimal_z:.2f}, confidence={confidence:.3f}, net_benefit={benefit - cost:.2f}"
        )

        return allocation

    def _calculate_optimal_allocation(
        self,
        initial_precision: float,
        value_difference: float,
        variance: float,
        importance: float,
        max_resources: float,
    ) -> float:
        """
        Calculate optimal resource allocation using gradient descent.

        Args:
            initial_precision: Initial precision (1/σ₀)
            value_difference: Value mode difference
            variance: Variance of value representations
            importance: Importance weight (R)
            max_resources: Maximum available resources

        Returns:
            Optimal resource amount
        """
        # Use grid search for optimization (simple but effective)
        best_z = 0.0
        best_net_benefit = float("-inf")

        # Search from 0 to max_resources
        for z in np.linspace(0, max_resources, 100):
            # Update precision
            precision = update_precision(initial_precision, z, self.config.beta)

            # Calculate confidence
            conf = confidence_from_delta_mu(
                value_difference, 1.0 / precision, self.config.lambda_conf
            )

            # Calculate net benefit
            benefit = calculate_benefit(z, importance, conf)
            cost = calculate_cost(z, self.config.alpha, self.config.nu)
            net_benefit = benefit - cost

            if net_benefit > best_net_benefit:
                best_net_benefit = net_benefit
                best_z = z

        return best_z

    def should_stop(
        self, current_score: float, iteration: int, allocation: Dict, target_score: float = 0.8
    ) -> bool:
        """
        Determine if optimal stopping criterion is met.

        Implements the control policy: Stop if success threshold reached OR resources exhausted.

        Args:
            current_score: Current quality score
            iteration: Current iteration number
            allocation: Current resource allocation
            target_score: Target score to achieve (pass_threshold)

        Returns:
            True if should stop, False if should continue
        """
        # 1. Success Criterion
        if current_score >= target_score:
            self.logger.info(
                f"Stopping: Target score reached ({current_score:.3f} >= {target_score})"
            )
            return True

        # 2. Resource Exhaustion Criterion
        if self.total_resources_allocated >= self.config.max_resources:
            self.logger.info(
                f"Stopping: Resources exhausted ({self.total_resources_allocated:.1f} >= {self.config.max_resources})"
            )
            return True

        # 3. Goal Completion Criterion
        if self.goal_stack:
            current_goal = self.goal_stack[-1]
            if current_goal.status == "completed":
                self.logger.debug(f"Stopping: Goal '{current_goal.description}' completed")
                return True

        # 4. Q-Value / Economic Stopping (Optional / Secondary)
        # We relax this to prioritize persistence unless cost is overwhelming
        # confidence = allocation["confidence"]
        # resources_used = (iteration + 1) * self.config.kappa
        # q_stop = self.config.R * confidence - self.config.alpha * (resources_used**self.config.nu)

        # Only stop on Q-value if we are VERY confident but somehow score is low (unlikely but possible)
        # OR if we are bleeding resources with no gain.
        # For now, we enforce persistence: don't stop just because of Q-value unless it's extreme.

        return False

    def establish_goal(
        self, description: str, priority: float = 0.5, parent_id: Optional[str] = None
    ) -> Goal:
        """
        Establish a new reasoning goal.

        Args:
            description: Goal description
            priority: Priority level (0.0-1.0)
            parent_id: Optional parent goal ID

        Returns:
            Created Goal object
        """
        goal_id = str(uuid.uuid4())
        goal = Goal(id=goal_id, description=description, priority=priority, parent_id=parent_id)

        self.goal_stack.append(goal)
        self.logger.info(f"Established goal: {description} (ID: {goal_id})")

        # Link to parent if exists
        if parent_id:
            for g in self.goal_stack:
                if g.id == parent_id:
                    g.subgoals.append(goal_id)
                    break

        return goal

    def adjust_goal(self, goal_id: str, modifications: Dict) -> Optional[Goal]:
        """
        Adjust an existing goal.

        Args:
            goal_id: ID of goal to adjust
            modifications: Dictionary of fields to update

        Returns:
            Updated Goal object or None if not found
        """
        for goal in self.goal_stack:
            if goal.id == goal_id:
                for key, value in modifications.items():
                    if hasattr(goal, key):
                        setattr(goal, key, value)
                self.logger.info(f"Adjusted goal {goal_id}: {modifications}")
                return goal
        return None

    def complete_goal(self, goal_id: str) -> bool:
        """
        Mark a goal as completed.

        Args:
            goal_id: ID of goal to complete

        Returns:
            True if successful
        """
        goal = self.adjust_goal(goal_id, {"status": "completed", "progress": 100.0})
        if goal:
            # If it's the current active goal (top of stack), pop it?
            # Or keep in history? For now, we keep it but status is completed.
            return True
        return False

    def get_current_goal(self) -> Optional[Goal]:
        """Get the current active goal (top of stack)."""
        # Return last non-completed goal
        for goal in reversed(self.goal_stack):
            if goal.status == "active":
                return goal
        return None

    def get_confidence_trajectory(self) -> list:
        """
        Get confidence values over decision history.

        Returns:
            List of confidence values
        """
        return [d["confidence"] for d in self.decision_history]

    def get_resource_trajectory(self) -> list:
        """
        Get resource allocation over decision history.

        Returns:
            List of resource amounts
        """
        return [d["amount"] for d in self.decision_history]

    def reset(self) -> None:
        """Reset controller state."""
        self.current_confidence = 0.5
        self.total_resources_allocated = 0.0
        self.decision_history.clear()
        self.logger.debug("oMCD controller reset")
