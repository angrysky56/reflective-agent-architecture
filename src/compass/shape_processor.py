"""
SHAPE Processor - Shorthand Assisted Prompt Engineering

Handles user input processing, shorthand expansion, semantic mapping,
and adaptive learning from feedback.
"""

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .config import SHAPEConfig
from .utils import COMPASSLogger


class SHAPEProcessor:
    """
    SHAPE: Shorthand Assisted Prompt Engineering

    Processes user input with shorthand expansion, semantic enrichment,
    and continuous adaptation based on feedback.
    """

    def __init__(
        self,
        config: SHAPEConfig,
        logger: Optional[COMPASSLogger] = None,
        llm_provider: Optional[Any] = None,
    ):
        """
        Initialize SHAPE processor.

        Args:
            config: SHAPEConfig instance
            logger: Optional logger instance
            llm_provider: Optional LLM provider
        """
        self.config = config
        self.logger = logger or COMPASSLogger("SHAPE")
        self.llm_provider = llm_provider

        # Shorthand dictionary
        self.shorthand_dict = dict(self.config.shorthand_dict)

        # Usage statistics for adaptation
        self.shorthand_usage: Dict[str, int] = defaultdict(int)
        self.expansion_feedback: List[Dict[str, Any]] = []

        self.logger.info("SHAPE processor initialized")

    async def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        Process user's prompt request and identify shorthand elements.
        Uses LLM if available for deeper semantic analysis.

        Args:
            user_input: Raw user input

        Returns:
            Processed input dictionary with identified shorthands
        """
        self.logger.debug(f"Processing user input: {user_input[:100]}...")

        # Tokenize input
        tokens = self._tokenize(user_input)

        # Identify shorthand elements (Heuristic)
        shorthand_elements = []
        for i, token in enumerate(tokens):
            if token.lower() in self.shorthand_dict:
                shorthand_elements.append(
                    {
                        "position": i,
                        "shorthand": token,
                        "expansion": self.shorthand_dict[token.lower()],
                        "context": self._get_context(tokens, i),
                    }
                )
                self.shorthand_usage[token.lower()] += 1

        result = {
            "original": user_input,
            "tokens": tokens,
            "shorthand_elements": shorthand_elements,
            "has_shorthand": len(shorthand_elements) > 0,
        }

        # LLM Semantic Enrichment (The "Intelligent" Part)
        if self.llm_provider:
            try:
                from .adapters import Message

                prompt = f"""Analyze the following user input for the SHAPE (Semantic Heuristic Analysis & Processing Engine) module.
Input: "{user_input}"

Identify:
1. Core Intent (What do they want to do?)
2. Key Entities (What objects/files/concepts are involved?)
3. Constraints (What are the limitations?)
4. Implicit Goals (What is the underlying objective?)

Return JSON: {{ "intent": "string", "entities": ["list"], "constraints": ["list"], "goals": ["list"] }}"""

                messages = [
                    Message(role="system", content="You are the SHAPE analysis module."),
                    Message(role="user", content=prompt),
                ]

                response_content = ""
                async for chunk in self.llm_provider.chat_completion(
                    messages, stream=False, temperature=0.2
                ):
                    response_content += chunk

                from .utils import extract_json_from_text

                analysis = extract_json_from_text(response_content)

                if not analysis:
                    # Provide fallback structure instead of raising
                    self.logger.warning("JSON extraction failed, using fallback heuristic analysis")
                    analysis = {
                        "intent": self._extract_intent(user_input),
                        "entities": self._extract_entities(user_input),
                        "constraints": self._extract_constraints(user_input),
                        "goals": self._extract_goals(user_input),
                    }

                result["llm_analysis"] = analysis
                self.logger.info(f"SHAPE LLM Analysis: {analysis.get('intent')}")

            except Exception as e:
                self.logger.warning(f"SHAPE LLM analysis failed: {e}")
                # Still provide fallback to prevent None
                result["llm_analysis"] = {
                    "intent": self._extract_intent(user_input),
                    "entities": self._extract_entities(user_input),
                    "constraints": self._extract_constraints(user_input),
                    "goals": self._extract_goals(user_input),
                }

        self.logger.debug(f"Found {len(shorthand_elements)} shorthand elements")
        return result

    def expand_shorthand(self, processed_input: Dict) -> str:
        """
        Expand shorthand to full form while maintaining context.

        Args:
            processed_input: Output from process_user_input

        Returns:
            Expanded prompt string
        """
        if not processed_input["has_shorthand"]:
            return str(processed_input["original"])

        tokens = processed_input["tokens"].copy()

        # Replace shorthands with expansions (in reverse order to maintain indices)
        for element in reversed(processed_input["shorthand_elements"]):
            pos = element["position"]
            expansion = element["expansion"]

            # Context-aware expansion if ML is enabled
            if self.config.enable_ml_expansion:
                expansion = self._context_aware_expansion(element["shorthand"], element["context"])

            tokens[pos] = expansion

        expanded = " ".join(tokens)
        self.logger.debug(f"Expanded: {expanded[:100]}...")

        return expanded

    def optimize_agent_prompt(
        self, task: str, context: Optional[Dict] = None, agent_role: str = "Assistant"
    ) -> str:
        """
        Optimize a raw task prompt for a specific agent role using SHAPE.

        Args:
            task: The raw task description
            context: Optional execution context
            agent_role: The role/persona of the agent (e.g., "Critic", "Creative")

        Returns:
            Optimized prompt string
        """
        context = context or {}

        # 1. Expand standard shorthands (e.g., "opt" -> "optimize")
        # We wrap the task in a dummy processed input structure to reuse expand_shorthand
        # In a real v2, we'd refactor this.
        processed: Dict[str, Any] = {
            "original": task,
            "tokens": self._tokenize(task),
            "shorthand_elements": [],
            "has_shorthand": False,
        }

        # Quick scan for shorthands
        # (Reusing logic from process_user_input but simplified for speed)
        tokens: List[str] = processed["tokens"]
        found_shorthand = False
        for i, token in enumerate(tokens):
            if token.lower() in self.shorthand_dict:
                elements: List[Dict[str, Any]] = processed["shorthand_elements"]
                elements.append(
                    {
                        "position": i,
                        "shorthand": token,
                        "expansion": self.shorthand_dict[token.lower()],
                        "context": {},
                    }
                )
                found_shorthand = True
        processed["has_shorthand"] = found_shorthand

        expanded_task = self.expand_shorthand(processed)

        # 2. Semantic Analysis
        semantics = {
            "intent": self._extract_intent(expanded_task),
            "entities": self._extract_entities(expanded_task),
            "constraints": self._extract_constraints(expanded_task),
            "goals": self._extract_goals(expanded_task),
        }

        # 3. Protocol Enforcement (The "Shape" of the prompt)
        # We inject specific instructions based on role and past feedback

        protocol_instructions = []

        # Base protocol: JSON output is mandatory
        protocol_instructions.append(
            "CRITICAL: Return your response in PURE JSON format (no markdown)."
        )

        # Role-based optimization
        if "critic" in agent_role.lower():
            protocol_instructions.append("Adopt a critical, skeptical perspective. Look for flaws.")
        elif "creative" in agent_role.lower():
            protocol_instructions.append("Think divergently. Explore novel connections.")

        # Feedback-based Adaptation (Validation loop)
        # If this type of task has failed recently, add safeguards.
        # Simple heuristic: if we have negative feedback for similar tasks
        recent_failures = [f for f in self.expansion_feedback if f["score"] < 0.5]
        if recent_failures and len(recent_failures) > 2:
            protocol_instructions.append("Think step-by-step before answering.")

        # 4. Construct Final Prompt
        optimized_prompt = f"""
System Role: {agent_role}

Task: {expanded_task}

Context Entities: {', '.join(semantics['entities'][:5])}
Implicit Goal: {', '.join(semantics['goals'][:2])}

Directives:
{chr(10).join([f'- {i}' for i in protocol_instructions])}
"""
        return optimized_prompt

    def map_semantics(self, expanded_prompt: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Refine expanded prompts for AI semantic alignment.

        Args:
            expanded_prompt: Expanded prompt from expand_shorthand
            context: Optional additional context

        Returns:
            Semantically optimized prompt dictionary
        """
        context = context or {}

        # Extract semantic elements
        semantic_elements = {
            "intent": self._extract_intent(expanded_prompt),
            "entities": self._extract_entities(expanded_prompt),
            "constraints": self._extract_constraints(expanded_prompt),
            "goals": self._extract_goals(expanded_prompt),
            "context": context,
        }

        # Build semantically rich representation
        result = {
            "prompt": expanded_prompt,
            "semantic_elements": semantic_elements,
            "enriched_prompt": self._build_enriched_prompt(expanded_prompt, semantic_elements),
        }

        self.logger.debug(f"Semantic mapping complete: intent={semantic_elements['intent']}")
        return result

    def collect_feedback(self, original_input: str, processed_output: Any, score: float) -> None:
        """
        Collect feedback for adaptation and evolution.

        Args:
            original_input: Original user input
            processed_output: System output
            score: Quality score
        """
        feedback_entry = {"input": original_input, "output": processed_output, "score": score}

        self.expansion_feedback.append(feedback_entry)

        # Adapt if we have enough feedback
        if len(self.expansion_feedback) >= 10:
            self._adapt_shorthand_dictionary()

    def _adapt_shorthand_dictionary(self) -> None:
        """
        Update shorthand dictionary based on feedback.

        This implements the Adaptation and Evolution (AE) step from SHAPE.
        """
        self.logger.info("Adapting shorthand dictionary based on feedback")

        # Analyze feedback
        high_score_inputs = [f for f in self.expansion_feedback if f["score"] > 0.8]

        # Extract potential new shorthands from high-performing inputs
        # This is a placeholder for more sophisticated ML-based detection
        if self.config.enable_ml_expansion:
            self._ml_based_adaptation(high_score_inputs)

        # Clear old feedback (keep only recent)
        self.expansion_feedback = self.expansion_feedback[-20:]

    def _ml_based_adaptation(self, high_score_inputs: List[Dict]) -> None:
        """
        Machine learning based shorthand adaptation.

        Placeholder for future ML implementation.
        """
        # TODO: Implement ML-based shorthand discovery
        pass

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization by whitespace."""
        return text.split()

    def _get_context(self, tokens: List[str], position: int) -> Dict[str, List[str]]:
        """
        Get context window around a token.

        Args:
            tokens: List of all tokens
            position: Position of target token

        Returns:
            Context dictionary with before/after tokens
        """
        window = self.config.context_window_size

        return {
            "before": tokens[max(0, position - window) : position],
            "after": tokens[position + 1 : position + 1 + window],
        }

    def _context_aware_expansion(self, shorthand: str, context: Dict) -> str:
        """
        Perform context-aware expansion using ML.

        Placeholder for future implementation.
        """
        # For now, just use dictionary expansion
        return self.shorthand_dict.get(shorthand.lower(), shorthand)

    def _extract_intent(self, prompt: str) -> str:
        """
        Extract primary intent from prompt.

        Simple heuristic-based extraction.
        """
        # Look for action verbs
        action_verbs = [
            "create",
            "build",
            "implement",
            "design",
            "optimize",
            "analyze",
            "evaluate",
            "solve",
            "find",
            "calculate",
            "generate",
            "develop",
            "test",
            "debug",
            "refactor",
        ]

        prompt_lower = prompt.lower()
        for verb in action_verbs:
            if verb in prompt_lower:
                return verb

        return "process"  # Default intent

    def _extract_entities(self, prompt: str) -> List[str]:
        """Extract key entities (simplified)."""
        # Very simple: extract capitalized words and technical terms
        entities = []

        # Capitalized words (basic NER)
        capitalized = re.findall(r"\b[A-Z][a-z]+\b", prompt)
        entities.extend(capitalized)

        # Technical terms (words with underscores or camelCase)
        technical = re.findall(r"\b\w+_\w+\b|\b[a-z]+[A-Z]\w+\b", prompt)
        entities.extend(technical)

        return list(set(entities))

    def _extract_constraints(self, prompt: str) -> List[str]:
        """Extract constraints from prompt."""
        constraints = []

        # Look for constraint keywords
        constraint_patterns = [
            r"must \w+",
            r"should \w+",
            r"cannot \w+",
            r"within \w+",
            r"using \w+",
            r"without \w+",
        ]

        for pattern in constraint_patterns:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            constraints.extend(matches)

        return constraints

    def _extract_goals(self, prompt: str) -> List[str]:
        """Extract goals from prompt."""
        goals = []

        # Look for goal indicators
        goal_patterns = [r"goal is to \w+", r"objective is \w+", r"aim to \w+", r"in order to \w+"]

        for pattern in goal_patterns:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            goals.extend(matches)

        # If no explicit goals, the intent serves as the goal
        if not goals:
            intent = self._extract_intent(prompt)
            goals = [f"to {intent}"]

        return goals

    def _build_enriched_prompt(self, original_prompt: str, semantic_elements: Dict) -> str:
        """
        Build an enriched prompt with semantic information.

        Args:
            original_prompt: Original expanded prompt
            semantic_elements: Extracted semantic elements

        Returns:
            Enriched prompt string
        """
        # Build structured representation
        enrichment = []

        enrichment.append(f"[INTENT: {semantic_elements['intent']}]")

        if semantic_elements["goals"]:
            enrichment.append(f"[GOALS: {', '.join(semantic_elements['goals'])}]")

        if semantic_elements["entities"]:
            enrichment.append(f"[ENTITIES: {', '.join(semantic_elements['entities'][:5])}]")

        if semantic_elements["constraints"]:
            enrichment.append(f"[CONSTRAINTS: {', '.join(semantic_elements['constraints'][:3])}]")

        # Combine with original
        enriched = f"{' '.join(enrichment)}\n\n{original_prompt}"

        return enriched
