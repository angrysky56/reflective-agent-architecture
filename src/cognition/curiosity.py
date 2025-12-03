import logging
import random
from collections import deque
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class CuriosityModule:
    """
    Implements Intrinsic Motivation based on Schmidhuber's Compression Progress.

    Drives the agent to explore areas where it expects to learn the most (maximize compression progress).
    Also tracks 'Boredom' to trigger exploration when the system is stuck in a loop or inactive.
    """

    def __init__(self, workspace):
        self.workspace = workspace
        # Short-term memory of recent operations to detect loops/boredom
        self.recent_operations = deque(maxlen=20)
        self.boredom_threshold = 0.7
        self.current_boredom = 0.0

    def record_activity(self, activity_type: str, details: str):
        """Record an activity to update boredom state."""
        self.recent_operations.append((activity_type, details))
        self._update_boredom()

    def _update_boredom(self):
        """
        Calculate boredom based on repetition and inactivity.
        Simple heuristic: High repetition = High boredom.
        """
        if len(self.recent_operations) < 5:
            self.current_boredom = 0.0
            return

        # Check for repetition in recent operations
        ops = [op[0] for op in self.recent_operations]
        unique_ops = set(ops)
        diversity = len(unique_ops) / len(ops)

        # Low diversity = High boredom
        # If diversity is 1.0 (all different), boredom is 0.0
        # If diversity is 0.1 (mostly same), boredom is 0.9
        self.current_boredom = 1.0 - diversity

        logger.debug(f"Curiosity Level: Boredom={self.current_boredom:.2f}")

    def should_explore(self) -> bool:
        """Decide if the system should trigger autonomous exploration."""
        return self.current_boredom > self.boredom_threshold

    def propose_goal(self) -> Optional[str]:
        """
        Propose a new goal for the Dreamer based on 'interesting' gaps.
        Uses explore_for_utility to find candidates.
        """
        if not self.should_explore():
            return None

        logger.info("Boredom threshold exceeded. Proposing exploration goal...")

        # 1. Active Exploration: Find high-utility/low-compression nodes
        candidates = self.workspace.explore_for_utility(max_candidates=3)

        if not candidates:
            return "Explore random concept to break stagnation."

        # 2. Formulate a goal
        # Pick the top candidate
        target = candidates[0]
        return f"Investigate concept '{target['name']}' to improve compression."
