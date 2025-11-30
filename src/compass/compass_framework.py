"""
COMPASS Framework - Core Orchestrator

COMPASS: Cognitive Orchestration & Metacognitive Planning for Adaptive Semantic Systems

This is the main orchestrator that coordinates all subsystems:
- SHAPE (User Interface Layer)
- oMCD + Self-Discover (Metacognitive Layer)
- SLAP + SMART (Reasoning & Planning Layer)
- Integrated Intelligence (Execution Layer)
"""


from typing import Any, Dict, List, Optional

from .config import COMPASSConfig, get_config
from .constraint_governor import ConstraintGovernor
from .executive_controller import ExecutiveController
from .integrated_intelligence import IntegratedIntelligence
from .omcd_controller import oMCDController
from .procedural_toolkit import ProceduralToolkit
from .representation_selector import RepresentationSelector
from .self_discover_engine import SelfDiscoverEngine
from .shape_processor import SHAPEProcessor
from .slap_pipeline import SLAPPipeline
from .smart_planner import SMARTPlanner
from .utils import COMPASSLogger, Trajectory


class COMPASS:
    """
    Main orchestrator for the COMPASS cognitive architecture.

    Integrates all six frameworks into a unified decision-making and reasoning system.
    """

    def __init__(self, config: Optional[COMPASSConfig] = None, logger: Optional[COMPASSLogger] = None, llm_provider: Optional[Any] = None, mcp_client: Optional[Any] = None):
        """
        Initialize COMPASS framework.

        Args:
            config: Configuration object (uses default if None)
            logger: Optional logger instance
            llm_provider: Optional LLM provider instance
            mcp_client: Optional MCP client instance
        """
        self.config = config or get_config()

        # Load tool configuration from environment
        import os
        tools_enabled = os.getenv("COMPASS_TOOLS_ENABLED", "false").lower() == "true"
        self.config.intelligence.enable_tools = tools_enabled

        self.logger = logger or COMPASSLogger("COMPASS", level=self.config.log_level)
        self.llm_provider = llm_provider
        self.mcp_client = mcp_client

        # Initialize components
        # Pass LLM provider to SHAPE for intelligent analysis
        self.shape_processor = SHAPEProcessor(self.config.shape, self.logger, llm_provider=self.llm_provider)
        self.smart_planner = SMARTPlanner(self.config.smart, self.logger)
        self.slap_pipeline = SLAPPipeline(self.config.slap, self.logger, llm_provider=self.llm_provider)
        self.omcd_controller = oMCDController(self.config.omcd, self.logger)
        self.self_discover_engine = SelfDiscoverEngine(self.config.self_discover, self.logger, llm_provider=self.llm_provider)
        self.integrated_intelligence = IntegratedIntelligence(self.config.intelligence, self.logger, llm_provider=self.llm_provider, mcp_client=self.mcp_client)

        # New: Constraint Governor
        self.constraint_governor = ConstraintGovernor(self.config.cgra.governor, self.logger)

        # New: Executive Controller
        self.executive_controller = ExecutiveController(self.config.cgra.executive, self.config.omcd, self.config.self_discover, self.logger)

        # Helper components
        self.representation_selector = RepresentationSelector(self.config.cgra.workspace, self.logger)
        self.procedural_toolkit = ProceduralToolkit(self.config.cgra.toolkit, self.logger)

        self.logger.info("COMPASS framework initialized with all components")

    async def process_task(self, task_description: str, context: Optional[Dict] = None, max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """
        Process a task through the integrated cognitive framework.

        Args:
            task_description: The task to process
            context: Optional context information
            max_iterations: Optional override for max iterations

        Returns:
            Dictionary containing the result and processing details
        """
        try:
            self.logger.info(f"Processing task: {task_description[:50]}...")

            # 1. SHAPE Input Processing
            self.logger.info("Phase 1: SHAPE Input Processing")
            # Now async and LLM-powered
            shape_result = await self.shape_processor.process_user_input(task_description)
            task_text = shape_result.get("original", task_description)  # Use original text

            # 2. Constraint Governor Validation (Input)
            self.logger.info("Phase 2: Constraint Validation")
            if not self.constraint_governor.validate_step({"input": task_text}, {"phase": "input_validation"}):
                self.logger.warning("Input validation failed constraints")
                # Continue but note violation

            # 3. SMART Objectives
            self.logger.info("Phase 3: SMART Objectives")
            objectives = self.smart_planner.create_objectives_from_task(task_text, context)
            self.logger.info(f"Created {len(objectives)} SMART objectives")

            # 4. Executive Control Loop (replaces direct calls to Self-Discover/oMCD)
            self.logger.info("Phase 4: Executive Control Loop")

            # Initialize loop variables
            trajectory = Trajectory(steps=[])
            current_state = {
                "task": task_text,
                "context": context or {},
                "objectives": objectives,
                "iteration": 0,
                "resources_used": 0.0,
                "confidence": 0.5,
                "score": 0.0,
            }

            max_iter = max_iterations or self.config.self_discover.max_trials
            final_score = 0.0
            solution = ""
            i = 0

            for i in range(max_iter):
                self.logger.info(f"Iteration {i+1}/{max_iter}")
                current_state["iteration"] = i

                # A. Meta-Cognitive Coordination
                # Executive Controller decides strategy and resources
                control_decision = self.executive_controller.coordinate_iteration(task_text, current_state, context)

                # Update state with decision
                current_state["strategy"] = control_decision["strategy"]
                current_state["resources"] = control_decision["resources"]

                # B. Representation Selection
                # Dynamic Workspace selects optimal representation
                representation = self.representation_selector.select_representation(task_text, context)
                self.logger.info(f"Selected representation: {representation.current_type} ({representation.confidence:.2f}) - {representation.reason_for_selection}")

                # C. Reasoning Plan Generation
                # SLAP generates plan based on selected representation
                self.logger.info(f"Creating SLAP reasoning plan (Type: {representation.current_type})")
                reasoning_plan = await self.slap_pipeline.create_reasoning_plan(task_text, objectives, representation_type=representation.current_type)
                self.logger.info(f"SLAP plan created with advancement score: {reasoning_plan.get('advancement', 0.0):.3f}")

                # D. Procedural Operations (Optional)
                # If Causal representation, use Backward Chaining from Toolkit
                if representation.current_type == "causal":
                    self.logger.info("Applying Causal Backward Chaining")
                    # Simplified goal extraction for MVP
                    goal = task_text
                    facts = context.get("facts", []) if context else []
                    chain = self.procedural_toolkit.backward_chaining(goal, facts)
                    # Enhance reasoning plan with chain
                    reasoning_plan["causal_chain"] = chain

                # E. Execution
                # Integrated Intelligence executes the plan

                # Enrich context with full cognitive state
                execution_context = context.copy() if context else {}

                # Get constraint violations
                violation_report = self.constraint_governor.get_violation_report()

                execution_context.update(
                    {
                        "trajectory": trajectory.to_dict(),  # Full history of actions and results
                        "shape_analysis": shape_result,  # How input was perceived
                        "smart_objectives": [obj.to_dict() for obj in objectives],  # What we are trying to achieve
                        "constraint_violations": violation_report,  # Self-critique/scrutiny
                        "iteration": i,
                    }
                )

                decision = await self._execute_reasoning_step(
                    task_text,
                    reasoning_plan,
                    control_decision["strategy"],  # strategy contains the module list
                    control_decision["resources"],
                    execution_context,  # Updated context with history
                )
                solution = decision.get("action", "")
                score = decision.get("confidence", 0.0)
                final_score = score
                current_state["score"] = score

                # Update trajectory
                trajectory.add_step(reasoning_plan, decision)

                # F. Evaluation & Reflection
                # Executive Controller evaluates quality
                evaluation = self.executive_controller.evaluate_reasoning_quality(trajectory, objectives)
                self.logger.info(f"Reasoning quality evaluation score: {evaluation}")

                # G. Goal Status Update
                if control_decision.get("goal"):
                    self.executive_controller.update_goal_status(control_decision["goal"].id, progress=score * 100)

                # Update SMART objectives progress for next iteration
                for obj in objectives:
                    obj.current_value = score * obj.target_value

                # Check stopping conditions
                if score >= self.config.self_discover.pass_threshold:
                    self.logger.info("Success threshold reached")
                    break

                if control_decision["should_stop"]:
                    self.logger.info("Executive Controller signaled stop")
                    break

            self.logger.info(f"Task completed: score={final_score:.3f}")

            return {
                "success": final_score >= self.config.self_discover.pass_threshold,
                "solution": solution,
                "score": final_score,
                "iterations": i + 1,
                "resources_used": self.omcd_controller.total_resources_allocated,
                "reflections": [r.to_dict() for r in self.self_discover_engine.get_memory()],
                "trajectory": trajectory.to_dict(),
            }

        except Exception as e:
            self.logger.error(f"Critical error in COMPASS process_task: {e}", exc_info=True)
            return {
                "success": False,
                "solution": f"Error: {str(e)}",
                "score": 0.0,
                "iterations": 0,
                "resources_used": 0.0,
                "reflections": [],
                "trajectory": {"steps": [{"error": str(e)}]}
            }

    async def _execute_reasoning_step(self, task: str, plan: Dict, modules: List[int], resources: Dict, context: Optional[Dict]) -> Dict:
        """Execute a single reasoning step using Integrated Intelligence."""
        self.logger.info(f"COMPASS._execute_reasoning_step called for task: {task[:50]}...")
        result = await self.integrated_intelligence.make_decision(task, plan, modules, resources, context or {})
        self.logger.info(f"COMPASS._execute_reasoning_step completed with action: {result.get('action', 'N/A')[:100]}...")
        return result

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the framework."""
        return {
            "omcd": {
                "confidence": self.omcd_controller.current_confidence,
                "resources_allocated": self.omcd_controller.total_resources_allocated,
            },
            "self_discover": {
                "memory_size": len(self.self_discover_engine.memory),
            },
            "task_history": [t.to_dict() for t in self.constraint_governor.reasoning_history],
        }


def create_compass(config: Optional[COMPASSConfig] = None, llm_provider: Optional[Any] = None, mcp_client: Optional[Any] = None) -> COMPASS:
    """
    Factory function to create a COMPASS instance.

    Args:
        config: Optional configuration override
        llm_provider: Optional LLM provider instance
        mcp_client: Optional MCP client instance

    Returns:
        Configured COMPASS instance
    """
    if config is None:
        config = COMPASSConfig()

    return COMPASS(config, llm_provider=llm_provider, mcp_client=mcp_client)


def quick_solve(task: str, **config_overrides) -> Dict[str, Any]:
    """
    Quick one-shot task solving with COMPASS.

    Args:
        task: Task description
        **config_overrides: Configuration overrides

    Returns:
        Solution result
    """
    from .config import create_custom_config

    config = create_custom_config(**config_overrides) if config_overrides else None
    compass = create_compass(config)

    return compass.process_task(task)
