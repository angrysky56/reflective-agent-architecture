"""
Self-Discover Engine - Reinforcement via Self-Reflection

Implements the Self-Discover framework with actor-evaluator-reflection loop,
reasoning module selection, and continuous improvement through self-evaluation.
"""

import random
from typing import Any, Dict, List, Optional

from .config import SelfDiscoverConfig
from .utils import COMPASSLogger, SelfReflection, Trajectory

# Reasoning modules from self_discover_TyMod.txt
REASONING_MODULES = {
    1: "How could I devise an experiment to help solve that problem?",
    2: "Make a list of ideas for solving this problem, and apply them one by one",
    3: "How could I measure progress on this problem?",
    4: "How can I simplify the problem so that it is easier to solve?",
    5: "What are the key assumptions underlying this problem?",
    6: "What are the potential risks and drawbacks of each solution?",
    7: "What are the alternative perspectives or viewpoints on this problem?",
    8: "What are the long-term implications of this problem and its solutions?",
    9: "How can I break down this problem into smaller, more manageable parts?",
    10: "Critical Thinking: Analyze from different perspectives, question assumptions",
    11: "Creative Thinking: Generate innovative and out-of-the-box ideas",
    12: "Collaborative Thinking: Seek input and leverage diverse perspectives",
    13: "Systems Thinking: Consider the problem as part of a larger system",
    14: "Risk Analysis: Evaluate risks, uncertainties, and tradeoffs",
    15: "Reflective Thinking: Examine biases and learn from past experiences",
    16: "What is the core issue or problem that needs to be addressed?",
    17: "What are the underlying causes or factors contributing to the problem?",
    18: "Are there any potential solutions that have been tried before?",
    19: "What are the potential obstacles or challenges that might arise?",
    20: "What data or information can provide insights into the problem?",
    21: "What are stakeholders' perspectives and needs?",
    22: "What resources are needed to tackle the problem effectively?",
    23: "How can progress or success be measured or evaluated?",
    24: "What indicators or metrics can be used?",
    25: "Is this a technical problem requiring specific expertise?",
    26: "Does the problem involve a physical constraint?",
    27: "Is the problem related to human behavior issues?",
    28: "Does this involve decision-making under uncertainty?",
    29: "Is this an analytical problem requiring data analysis?",
    30: "Is this a design challenge requiring creative solutions?",
    31: "Does this require addressing systemic or structural issues?",
    32: "Is this problem time-sensitive or urgent?",
    33: "What kinds of solutions typically work for this problem type?",
    34: "What are other possible solutions given the current best solution?",
    35: "What if the current best solution is totally wrong?",
    36: "How to modify the current solution based on problem characteristics?",
    37: "Ignoring current solution, create an entirely new solution",
    38: "Let's think step by step",
    39: "Let's make a step-by-step plan and implement it",
}


class SelfDiscoverEngine:
    """
    Self-Discover: Reinforcement via Self-Reflection

    Manages the actor-evaluator-reflection loop for continuous improvement
    through self-evaluation and adaptation.
    """

    def __init__(self, config: SelfDiscoverConfig, logger: Optional[COMPASSLogger] = None, llm_provider: Optional[Any] = None):
        """
        Initialize Self-Discover Engine.

        Args:
            config: Configuration object
            logger: Optional logger instance
            llm_provider: Optional LLM provider instance
        """
        self.config = config
        self.logger = logger or COMPASSLogger("SelfDiscover")
        self.llm_provider = llm_provider

        # Module effectiveness tracking
        self.module_effectiveness = {i: 0.5 for i in REASONING_MODULES.keys()}

        # Memory of past reflections
        self.memory: List[SelfReflection] = []

        self.logger.info("Self-Discover engine initialized")

    def select_reasoning_modules(self, task: str, reflections: List[SelfReflection], context: Optional[Dict] = None) -> List[int]:
        """
        Select the best reasoning modules for the given task.

        Implements module selection based on strategy and past effectiveness.

        Args:
            task: Task description
            reflections: Past self-reflections
            context: Optional context dictionary

        Returns:
            List of selected module indices
        """
        strategy = self.config.module_selection_strategy

        if strategy == "all":
            selected = self.config.enabled_reasoning_modules

        elif strategy == "random":
            k = min(self.config.top_k_modules, len(self.config.enabled_reasoning_modules))
            selected = random.sample(self.config.enabled_reasoning_modules, k)

        elif strategy == "top_k":
            # Select top K by effectiveness
            sorted_modules = sorted(self.module_effectiveness.items(), key=lambda x: x[1], reverse=True)
            selected = [m[0] for m in sorted_modules[: self.config.top_k_modules]]

        elif strategy == "adaptive":
            # Adaptive selection based on task and past performance
            selected = self._adaptive_selection(task, reflections, context)

        else:
            raise ValueError(f"Unknown module selection strategy: {strategy}")

        self.logger.debug(f"Selected modules: {selected}")
        return selected

    def _adaptive_selection(self, task: str, reflections: List[SelfReflection], context: Optional[Dict] = None) -> List[int]:
        """
        Adaptively select modules based on task characteristics and past performance.

        Args:
            task: Task description
            reflections: Past reflections
            context: Optional context dictionary

        Returns:
            Selected module indices
        """
        # Categorize task
        task_lower = task.lower()

        # Build preference weights
        weights = dict(self.module_effectiveness)

        # Boost certain modules based on task type
        if any(word in task_lower for word in ["create", "design", "build"]):
            # Creative/design task
            for module in [11, 30, 37]:  # Creative thinking, design challenge
                weights[module] = weights.get(module, 0.5) * 1.5

        if any(word in task_lower for word in ["analyze", "evaluate", "assess"]):
            # Analytical task
            for module in [10, 14, 29]:  # Critical thinking, risk analysis
                weights[module] = weights.get(module, 0.5) * 1.5

        if any(word in task_lower for word in ["optimize", "improve", "enhance"]):
            # Optimization task
            for module in [4, 13, 36]:  # Simplify, systems thinking
                weights[module] = weights.get(module, 0.5) * 1.5

        if "quick" in task_lower or "urgent" in task_lower:
            weights[32] = weights.get(32, 0.5) * 2.0  # Time-sensitive

        # Context-aware boosting
        if context:
            if context.get("complexity") == "high":
                weights[9] = weights.get(9, 0.5) * 1.5  # Decomposition
                weights[13] = weights.get(13, 0.5) * 1.5  # Systems thinking

            if context.get("uncertainty") == "high":
                weights[14] = weights.get(14, 0.5) * 1.5  # Risk analysis
                weights[28] = weights.get(28, 0.5) * 1.5  # Decision under uncertainty

            if context.get("domain") == "creative":
                weights[11] = weights.get(11, 0.5) * 1.5  # Creative thinking

        # Consider recent reflection insights
        if reflections:
            recent_reflections = reflections[-3:]
            for reflection in recent_reflections:
                # Boost modules mentioned in successful reflections
                for insight in reflection.insights:
                    for module_id, description in REASONING_MODULES.items():
                        if any(word in insight.lower() for word in description.lower().split()[:3]):
                            weights[module_id] = weights.get(module_id, 0.5) * 1.2

        # Sample based on weights (epsilon-greedy)
        epsilon = 0.2  # 20% exploration

        if random.random() < epsilon:
            # Explore: random selection
            k = min(self.config.top_k_modules, len(self.config.enabled_reasoning_modules))
            selected = random.sample(self.config.enabled_reasoning_modules, k)
        else:
            # Exploit: select by weight
            sorted_modules = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            selected = [m[0] for m in sorted_modules[: self.config.top_k_modules]]

        return selected

    def evaluate_trajectory(self, trajectory: Trajectory, objectives: List) -> float:
        """
        Evaluate a trajectory's quality.

        Args:
            trajectory: Trajectory to evaluate
            objectives: List of objectives

        Returns:
            Score in [0, 1]
        """
        if len(trajectory) == 0:
            return 0.0

        # Simple heuristic evaluation (can be enhanced)
        # - Longer trajectories with progress get higher scores
        # - Completion of objectives increases score

        base_score = min(0.5, 0.4 + len(trajectory) * 0.05)  # Base confidence for active trajectory

        # Check if objectives are met
        if objectives:
            objectives_met = sum(1 for obj in objectives if hasattr(obj, "is_on_track") and obj.is_on_track)
            objective_score = objectives_met / len(objectives) * 0.5
        else:
            objective_score = 0.3  # Default if no objectives

        total_score = base_score + objective_score

        # Boost score if LLM provider is present (assuming higher quality reasoning)
        if self.llm_provider:
            total_score = min(1.0, total_score * 1.2)

        self.logger.debug(f"Trajectory score: {total_score:.3f}")
        return min(1.0, total_score)

    def generate_reflection(self, trajectory: Trajectory, score: float, objectives: List) -> SelfReflection:
        """
        Generate self-reflection on trajectory performance.

        Args:
            trajectory: Trajectory to reflect on
            score: Trajectory score
            objectives: List of objectives

        Returns:
            SelfReflection object
        """
        # Analyze trajectory
        insights = self._extract_insights(trajectory, score, objectives)
        improvements = self._identify_improvements(trajectory, score, insights)

        # Generate reflection content
        content = self._build_reflection_content(trajectory, score, insights, improvements)

        # Assess context awareness
        context_awareness = self._assess_context_awareness(trajectory, objectives)

        reflection = SelfReflection(trajectory_id=len(self.memory), content=content, insights=insights, improvements=improvements, context_awareness=context_awareness)

        # Update memory
        self.memory.append(reflection)
        if len(self.memory) > self.config.max_memory_size:
            self.memory.pop(0)

        # Update module effectiveness based on reflection
        self._update_module_effectiveness(insights, score)

        self.logger.debug(f"Generated reflection with {len(insights)} insights")
        return reflection

    def assess_task_solvability(self, task: str) -> Dict[str, Any]:
        """
        Assess the solvability of a task (Self-Awareness).

        Args:
            task: Task description

        Returns:
            Dictionary with solvability assessment
        """
        # Heuristic assessment
        complexity_score = 0.5
        if len(task.split()) > 50:
            complexity_score += 0.2
        if any(word in task.lower() for word in ["impossible", "unknown", "undecidable"]):
            complexity_score += 0.3

        return {"solvability_score": max(0.0, 1.0 - complexity_score), "estimated_complexity": complexity_score, "missing_info_detected": "?" in task or "find" in task.lower()}

    def _assess_context_awareness(self, trajectory: Trajectory, objectives: List) -> Dict[str, Any]:
        """Assess context awareness based on trajectory."""
        return {"adaptability_score": 0.5 + (0.1 * len(trajectory) if len(trajectory) < 10 else -0.1), "objective_alignment": sum(1 for obj in objectives if obj.is_on_track) / len(objectives) if objectives else 0.0, "resource_efficiency": 1.0 / (len(trajectory) + 1)}

    def _extract_insights(self, trajectory: Trajectory, score: float, objectives: List) -> List[str]:
        """Extract insights from trajectory."""
        insights = []

        if score > 0.8:
            insights.append("High-quality solution achieved")
            insights.append("Effective reasoning approach used")
        elif score > 0.5:
            insights.append("Moderate success, room for improvement")
        else:
            insights.append("Low performance, need different approach")

        if len(trajectory) > 8:
            insights.append("Trajectory was lengthy, consider more efficient paths")
        elif len(trajectory) < 3:
            insights.append("Solution found quickly, possibly oversimplified")

        return insights

    def _identify_improvements(self, trajectory: Trajectory, score: float, insights: List[str]) -> List[str]:
        """Identify potential improvements."""
        improvements = []

        if score < 0.7:
            improvements.append("Consider different reasoning modules")
            improvements.append("Allocate more cognitive resources")

        if len(trajectory) > 10:
            improvements.append("Optimize for more direct solution paths")

        if "oversimplified" in str(insights):
            improvements.append("Increase scrutiny and validation steps")

        return improvements

    def _build_reflection_content(self, trajectory: Trajectory, score: float, insights: List[str], improvements: List[str]) -> str:
        """Build reflection content string."""
        content = f"Reflection on Trajectory {len(self.memory)} (Score: {score:.3f})\n\n"
        content += f"Trajectory length: {len(trajectory)} steps\n\n"
        content += "Insights:\n" + "\n".join(f"- {i}" for i in insights) + "\n\n"
        content += "Improvements:\n" + "\n".join(f"- {i}" for i in improvements)

        return content

    def _update_module_effectiveness(self, insights: List[str], score: float):
        """Update module effectiveness based on reflection."""
        # Simple update: increase effectiveness for all modules if score is high
        learning_rate = 0.1

        for module_id in self.module_effectiveness:
            # Move towards score with learning rate
            self.module_effectiveness[module_id] = (1 - learning_rate) * self.module_effectiveness[module_id] + learning_rate * score

    def get_memory(self) -> List[SelfReflection]:
        """Get reflection memory."""
        return self.memory.copy()

    def reset(self):
        """Reset engine state."""
        self.memory.clear()
        self.module_effectiveness = {i: 0.5 for i in REASONING_MODULES.keys()}
        self.logger.debug("Self-Discover engine reset")
