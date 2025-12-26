"""
Representation Selector - Dynamic Workspace Management

Implements the Representation Selector module of the CGRA architecture.
Analyzes task characteristics to select the optimal reasoning representation
(Sequential, Hierarchical, Network, Causal, etc.).
"""

from typing import Dict, Optional

from .config import DynamicWorkspaceConfig
from .utils import COMPASSLogger, RepresentationState


class RepresentationSelector:
    """
    Selects and manages reasoning representations based on task demands.
    """

    def __init__(self, config: DynamicWorkspaceConfig, logger: Optional[COMPASSLogger] = None):
        """
        Initialize Representation Selector.

        Args:
            config: Dynamic Workspace configuration
            logger: Optional logger
        """
        self.config = config
        self.logger = logger or COMPASSLogger("RepresentationSelector")
        self.logger.info("Representation Selector initialized")

    def select_representation(self, task: str, context: Optional[Dict] = None) -> RepresentationState:
        """
        Select the optimal representation for a task.

        Args:
            task: Task description
            context: Optional context

        Returns:
            RepresentationState object
        """
        task_lower = task.lower()
        context = context or {}

        # Default to configured default
        selected_type = self.config.default_representation
        confidence = 0.5
        reason = "Default selection"

        # 1. Causal Detection (Cause-Effect) - Specific
        if any(word in task_lower for word in ["cause", "effect", "why", "impact", "consequence", "root cause"]):
            selected_type = "causal"
            confidence = 0.85
            reason = "Task requires causal reasoning"

        # 2. Network Detection (Relationships) - Specific
        elif any(word in task_lower for word in ["connect", "relate", "network", "graph", "link", "map"]):
            selected_type = "network"
            confidence = 0.75
            reason = "Task involves complex relationships"

        # 3. Hierarchical Detection (Decomposition) - Generic
        elif any(word in task_lower for word in ["design", "system", "architecture", "break down", "decompose", "plan"]):
            selected_type = "hierarchical"
            confidence = 0.8
            reason = "Task requires structural decomposition"

        # 4. Spatial Detection
        elif any(word in task_lower for word in ["spatial", "layout", "position", "where", "map", "route"]):
            selected_type = "spatial"
            confidence = 0.9
            reason = "Task involves spatial relationships"

        # 5. Temporal Detection
        elif any(word in task_lower for word in ["timeline", "schedule", "history", "sequence", "when", "order"]):
            selected_type = "temporal"
            confidence = 0.8
            reason = "Task involves temporal ordering"

        # Check if allowed
        if selected_type not in self.config.allowed_representations:
            self.logger.warning(f"Selected type '{selected_type}' not allowed, falling back to default")
            selected_type = self.config.default_representation
            confidence = 0.5
            reason = "Fallback: Selected type not allowed"

        self.logger.info(f"Selected representation: {selected_type} ({confidence:.2f}) - {reason}")

        return RepresentationState(current_type=selected_type, confidence=confidence, history=[selected_type], reason_for_selection=reason)

    def should_switch_representation(self, current_state: RepresentationState, progress: float, step_data: Dict) -> Optional[str]:
        """
        Determine if representation should be switched mid-task.

        Args:
            current_state: Current representation state
            progress: Task progress (0.0-1.0)
            step_data: Data from last reasoning step

        Returns:
            New representation type or None
        """
        if not self.config.enable_adaptive_switching:
            return None

        # Heuristic: If stuck (low progress) and using sequential, try hierarchical
        if progress < 0.2 and len(current_state.history) > 5 and current_state.current_type == "sequential":
            return "hierarchical"

        # Heuristic: If discovering many relationships, switch to network
        content = str(step_data.get("reasoning", "")).lower()
        if "relationship" in content or "connect" in content:
            if current_state.current_type not in ["network", "causal"]:
                return "network"

        return None
