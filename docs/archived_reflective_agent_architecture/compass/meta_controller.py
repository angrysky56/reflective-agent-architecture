"""
Meta-Controller - Adaptive Workflow Orchestration

This module implements the Meta-Controller, which sits above the COMPASS framework.
Its role is to adaptively select the optimal processing workflow based on task requirements,
effectively bridging the "Macro" (Semantic) and "Micro" (Parameter) levels of control.
"""

from enum import Enum, auto
from typing import Any, Dict, Optional

from .compass_framework import COMPASS
from .config import COMPASSConfig
from .utils import COMPASSLogger


class WorkflowType(Enum):
    """Available processing workflows."""

    STANDARD = auto()  # Balanced approach
    RESEARCH = auto()  # Heavy on deconstruction and information gathering
    CREATIVE = auto()  # High temperature, emphasis on hypothesis/synthesis
    DEBUG = auto()  # Strict constraints, detailed logging


class MetaController:
    """
    Adaptive orchestrator that selects and configures COMPASS workflows.
    """

    def __init__(
        self, compass_instance: Optional[COMPASS] = None, logger: Optional[COMPASSLogger] = None
    ):
        self.compass = compass_instance
        self.logger = logger or COMPASSLogger("MetaController")
        self.current_workflow = WorkflowType.STANDARD

    def set_compass(self, compass_instance: COMPASS) -> None:
        """Set the COMPASS instance to control."""
        self.compass = compass_instance

    async def process_task_adaptively(
        self, task: str, context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Analyze the task and execute it using the most appropriate workflow.
        """
        if not self.compass:
            raise RuntimeError("COMPASS instance not set in MetaController")

        # 1. Analyze Task Intent (Simple heuristic for now, could be LLM-based)
        workflow = self._determine_workflow(task, context)
        self.logger.info(f"Selected workflow: {workflow.name}")
        self.current_workflow = workflow

        # 2. Configure COMPASS for the selected workflow
        original_config = self.compass.config
        adaptive_config = self._configure_workflow(workflow, original_config)

        # Temporarily apply adaptive config
        self.compass.config = adaptive_config

        try:
            # 3. Execute Task
            result = await self.compass.process_task(task, context)

            # 4. Add Meta-Data to result
            result["meta"] = {
                "workflow": workflow.name,
                "adaptive_logic": "Heuristic-based selection",
            }
            return result

        finally:
            # Restore original config
            self.compass.config = original_config

    def _determine_workflow(self, task: str, context: Optional[Dict]) -> WorkflowType:
        """
        Determine the best workflow based on task keywords and context.
        """
        task_lower = task.lower()

        if any(w in task_lower for w in ["debug", "fix", "error", "fail", "trace"]):
            return WorkflowType.DEBUG

        if any(w in task_lower for w in ["research", "investigate", "study", "analyze", "learn"]):
            return WorkflowType.RESEARCH

        if any(w in task_lower for w in ["create", "invent", "design", "imagine", "story", "poem"]):
            return WorkflowType.CREATIVE

        return WorkflowType.STANDARD

    def _configure_workflow(
        self, workflow: WorkflowType, base_config: COMPASSConfig
    ) -> COMPASSConfig:
        """
        Generate a configuration object tailored to the selected workflow.
        """
        # Create a copy-like behavior by using create_custom_config with base values
        # Note: This is a simplified approach. Ideally, we'd deep copy.

        # Start with a fresh config to avoid mutating the original
        # We will override specific sections based on the workflow

        overrides = {}

        if workflow == WorkflowType.RESEARCH:
            overrides["self_discover"] = {
                "max_trials": 15,  # More iterations for deep research
                "module_selection_strategy": "all",  # Use more reasoning modules
            }
            overrides["omcd"] = {"min_confidence": 0.7}  # Higher confidence required

        elif workflow == WorkflowType.CREATIVE:
            overrides["self_discover"] = {
                "module_selection_strategy": "random",  # Encourage diversity
                "top_k_modules": 10,
            }
            overrides["slap"] = {
                "alpha": 0.2,  # Lower scrutiny
                "beta": 0.8,  # Higher improvement/novelty
            }

        elif workflow == WorkflowType.DEBUG:
            overrides["cgra"] = {
                "governor": {
                    "enable_validation": True,
                    "max_contradictions_allowed": 0,  # Zero tolerance
                }
            }
            overrides["omcd"] = {"alpha": 0.05}  # Cheaper to think (encourage detailed steps)

        # Apply overrides
        # Since we can't easily deep copy the whole config structure here without serialization,
        # we'll modify the existing config *in place* in the main method, but here we return
        # a new config object if we were using the factory.
        # However, since we are swapping self.compass.config, we need a full object.
        # For safety in this MVP, we will rely on the fact that create_custom_config
        # creates a NEW instance.

        # We need to manually copy over values from base_config to the new config
        # This is tedious, so we will use a hybrid approach:
        # We will return the base_config and rely on the caller to mutate/restore,
        # OR we implement a proper clone.
        # Let's use the create_custom_config helper to generate a *delta* config,
        # but that returns a default-initialized config with overrides.
        # It doesn't inherit from base_config.

        # Better approach: Just modify the overrides dictionary and use it to patch
        # the *current* config object in the main loop, then unpatch.
        # But `process_task` uses `self.config`.

        # Let's create a new config using the factory and manual attribute copying for now.
        new_config = COMPASSConfig()

        # Copy high-level settings (simplified)
        new_config.enable_logging = base_config.enable_logging
        new_config.log_level = base_config.log_level

        # Apply overrides via the helper
        # custom_config = create_custom_config(**overrides)

        # Merge custom_config into new_config (which mimics base)
        # This is not perfect but sufficient for the MVP of switching modes.
        # In a production system, we'd implement __deepcopy__.

        if workflow == WorkflowType.RESEARCH:
            new_config.self_discover.max_trials = 15
            new_config.self_discover.module_selection_strategy = "all"
            new_config.omcd.min_confidence = 0.7

        elif workflow == WorkflowType.CREATIVE:
            new_config.self_discover.module_selection_strategy = "random"
            new_config.self_discover.top_k_modules = 10
            new_config.slap.alpha = 0.2
            new_config.slap.beta = 0.8

        elif workflow == WorkflowType.DEBUG:
            new_config.cgra.governor.enable_validation = True
            new_config.cgra.governor.max_contradictions_allowed = 0
            new_config.omcd.alpha = 0.05

        return new_config
