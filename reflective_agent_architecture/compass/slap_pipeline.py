"""
SLAP Pipeline - Semantic Logic Auto Progressor

Implements the SLAP framework's logical flow:
C(R(F(S(D(RB(M(SF))))))) - Conceptualization → Representation → Facts →
Scrutiny → Derivation → Rule-Based → Model → Semantic Formalization

Now powered by LLM for deep semantic reasoning and advancement tracking.
"""

import json
from typing import Any, Dict, List, Optional

from .adapters import Message
from .config import SLAPConfig
from .utils import COMPASSLogger, extract_json_from_text


class SLAPPipeline:
    """
    SLAP: Semantic Logic Auto Progressor

    Processes information through a structured semantic pipeline with
    continuous advancement tracking using LLM-driven reasoning.
    """

    def __init__(
        self,
        config: SLAPConfig,
        logger: Optional[COMPASSLogger] = None,
        llm_provider: Optional[Any] = None,
    ):
        """
        Initialize SLAP pipeline.

        Args:
            config: SLAPConfig instance
            logger: Optional logger instance
            llm_provider: LLM provider instance
        """
        self.config = config
        self.logger = logger or COMPASSLogger("SLAP")
        self.llm_provider = llm_provider

        # Pipeline state
        self.current_truth = 1.0
        self.current_scrutiny = 0.0
        self.current_improvement = 0.0
        self.advancement_history: List[float] = []

        self.logger.info("SLAP pipeline initialized with LLM support")

    async def create_reasoning_plan(
        self, task: str, objectives: List, representation_type: str = "sequential"
    ) -> Dict[str, Any]:
        """
        Create a structured reasoning plan using the SLAP pipeline via LLM.

        Processes through: C → R → F → S → D → RB → M → SF

        Args:
            task: Task description
            objectives: List of objectives
            representation_type: Type of representation

        Returns:
            Reasoning plan dictionary
        """
        self.logger.info(f"Creating SLAP reasoning plan (Type: {representation_type})")
        self.logger.info(
            f"SLAP llm_provider: {'SET' if self.llm_provider else 'None'} (type: {type(self.llm_provider).__name__ if self.llm_provider else 'N/A'})"
        )

        if not self.llm_provider:
            self.logger.warning("No LLM provider available for SLAP. Returning fallback plan.")
            return self._create_fallback_plan(task, representation_type)

        # Construct the SLAP Prompt
        system_prompt = """You are the SLAP (Semantic Logic Auto Progressor) engine of the COMPASS cognitive architecture.
Your goal is to process a task through 8 distinct semantic stages to build a robust reasoning plan.

The 8 Stages are:
1. Conceptualization (C): Identify core concepts, abstractions, and domains.
2. Representation (R): Map concepts to a structure (sequential, hierarchical, network, or causal).
3. Facts (F): Extract explicit and implicit facts from the representation.
4. Scrutiny (S): Critically analyze facts for gaps, weaknesses, and inconsistencies.
5. Derivation (D): Derive new insights or implications from the facts and scrutiny.
6. Rule-Based (RB): Apply logical rules (consolidation, validation, prioritization) to the derivations.
7. Model (M): Synthesize a structured mental model of the solution.
8. Semantic Formalization (SF): Formalize the model into an executable plan.

You must output a JSON object containing the analysis for each stage and a self-evaluated advancement score."""

        user_prompt = f"""Task: {task}
Representation Type: {representation_type}
Objectives: {[obj.description if hasattr(obj, "description") else str(obj) for obj in objectives]}

Perform the SLAP analysis and output valid JSON in this format:
{{
    "conceptualization": {{
        "primary_concept": "string",
        "related_concepts": ["string"],
        "abstract_level": "low|medium|high",
        "domain": "string"
    }},
    "representation": {{
        "type": "{representation_type}",
        "structure": "description of structure",
        "relationships": ["concept A relates to concept B"]
    }},
    "facts": [
        {{"type": "structural|relational|contextual", "statement": "fact statement", "confidence": 0.0-1.0}}
    ],
    "scrutiny": {{
        "weaknesses": ["string"],
        "gaps": ["string"],
        "inconsistencies": ["string"],
        "score": 0.0-1.0 (lower is better/less issues)
    }},
    "derivation": [
        {{"statement": "derived insight", "confidence": 0.0-1.0}}
    ],
    "rules": {{
        "applied_rules": ["rule name"],
        "outcomes": ["rule application result"]
    }},
    "model": {{
        "components": ["component 1", "component 2"],
        "structure": "{representation_type}",
        "completeness": 0.0-1.0
    }},
    "semantic": {{
        "formalized": true,
        "ready_for_execution": true|false,
        "execution_plan": ["step 1", "step 2"]
    }},
    "advancement_score": 0.0-3.0 (calculated as Truth + 0.4*Scrutiny + 0.6*Improvement)
}}"""

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]

        try:
            # Define tools_to_use for the LLM call (assuming no specific tools are needed for this SLAP plan generation)
            tools_to_use: List[Dict[str, Any]] = []

            # Call LLM
            response_content = ""
            self.logger.info("SLAP: About to call llm_provider.chat_completion")

            try:
                async for chunk in self.llm_provider.chat_completion(
                    messages, stream=False, temperature=0.3, max_tokens=16000, tools=tools_to_use
                ):
                    response_content += chunk
                self.logger.info(
                    f"SLAP: LLM call completed, response length: {len(response_content)}"
                )
            except Exception as iteration_error:
                self.logger.error(
                    f"SLAP: Error during LLM iteration: {iteration_error}", exc_info=True
                )

                raise  # Re-raise to trigger outer exception handler

            # Parse JSON
            try:
                # Parse JSON
                plan = extract_json_from_text(response_content)

                if not plan:
                    self.logger.warning(
                        f"SLAP: JSON extraction failed. Response preview: {response_content[:200]}"
                    )

                    raise json.JSONDecodeError("Failed to extract JSON", response_content, 0)

                # Ensure advancement score exists
                if "advancement_score" not in plan:
                    plan["advancement_score"] = 1.5  # Default fallback

                # Update history
                self.advancement_history.append(plan["advancement_score"])
                self.current_truth = 1.0 + (plan["model"]["completeness"] * 0.5)
                self.current_scrutiny = plan["scrutiny"]["score"]
                self.current_improvement = plan["model"]["completeness"]

                # Map to expected output format for compatibility
                plan["advancement"] = plan["advancement_score"]
                plan["type"] = representation_type

                self.logger.info(f"SLAP plan created via LLM with score: {plan['advancement']:.3f}")
                self.logger.info(f"SLAP_DETAILED_PLAN: {json.dumps(plan, indent=2)}")
                return plan

            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse SLAP LLM response: {e}")
                self.logger.debug(f"Raw response: {response_content}")
                return self._create_fallback_plan(task, representation_type)

        except Exception as e:
            self.logger.error(f"Error in SLAP LLM execution: {e}", exc_info=True)
            self.logger.error(f"SLAP llm_provider type: {type(self.llm_provider)}")
            self.logger.error(f"Exception details: {type(e).__name__}: {str(e)}")

            return self._create_fallback_plan(task, representation_type)

    def _create_fallback_plan(self, task: str, representation_type: str) -> Dict[str, Any]:
        """Create a basic fallback plan if LLM fails."""
        return {
            "type": representation_type,
            "conceptualization": {
                "primary_concept": "Task Processing",
                "related_concepts": ["Execution", "Analysis"],
                "domain": "general",
            },
            "representation": {"structure": "linear"},
            "facts": [],
            "scrutiny": {"gaps": ["LLM generation failed"], "score": 1.0},
            "model": {"completeness": 0.1},
            "semantic": {"ready_for_execution": False},
            "advancement": 0.5,
        }

    def identify_missing_entities_mcts(
        self, current_plan: Dict, iterations: Optional[int] = None
    ) -> List[str]:
        """
        Identify missing entities (simplified for now, could be LLM-enhanced later).
        """
        missing = []
        if "scrutiny" in current_plan and "gaps" in current_plan["scrutiny"]:
            missing.extend(current_plan["scrutiny"]["gaps"])
        return missing

    def reset(self) -> None:
        """Reset pipeline state."""
        self.current_truth = 1.0
        self.current_scrutiny = 0.0
        self.current_improvement = 0.0
        self.advancement_history.clear()
        self.logger.debug("SLAP pipeline reset")
