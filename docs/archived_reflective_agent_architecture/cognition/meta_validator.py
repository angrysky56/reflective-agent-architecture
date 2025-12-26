import math
import re
from typing import Any, Callable, Dict


class MetaValidator:
    """
    Implements the Unified Evaluation Ontology to reconcile Constraint Validator (Coverage)
    and Self-Critique (Rigor).
    """

    @staticmethod
    def compute_specificity(text: str) -> float:
        """
        Measure concrete instantiation via lexical analysis.
        """
        specificity_indicators = {
            # Formulas and equations
            "formula_count": len(re.findall(r"[=<>∈∀∃∫∑]", text)),
            # Concrete numbers and values
            "number_count": len(re.findall(r"\b\d+(\.\d+)?\b", text)),
            # Explicit examples
            "example_markers": len(
                re.findall(r"\b(e\.g\.|for example|such as|specifically)\b", text, re.I)
            ),
            # Defined thresholds
            "threshold_markers": len(re.findall(r"\b(≥|≤|<|>|threshold|cutoff)\b", text)),
            # Concrete names and terms (heuristic: capitalized words not at start of sentence)
            # This is a simple approximation
            "proper_nouns": len(re.findall(r"\b[A-Z][a-z]+\b", text)),
        }

        # Normalize by text length
        text_length = len(text.split())
        normalized_specificity = sum(specificity_indicators.values()) / max(text_length, 1)

        # Scale to [0, 1] (heuristic scaling factor)
        return min(normalized_specificity * 10, 1.0)

    @staticmethod
    def compute_justification_depth(text: str) -> float:
        """
        Measure argumentative structure via discourse markers.
        """
        premise_markers = len(
            re.findall(r"\b(given|assume|suppose|if|since|because)\b", text, re.I)
        )
        inference_markers = len(
            re.findall(r"\b(therefore|thus|hence|consequently|implies)\b", text, re.I)
        )
        conclusion_markers = len(
            re.findall(r"\b(conclude|follows|shows|proves|demonstrates)\b", text, re.I)
        )

        # Check for logical structure: premises -> inference -> conclusion
        has_structure = premise_markers > 0 and inference_markers > 0 and conclusion_markers > 0

        depth_score = (premise_markers + inference_markers + conclusion_markers) / max(
            len(text.split()), 1
        )

        # Boost if complete logical structure present
        if has_structure:
            depth_score *= 1.5

        return min(depth_score * 20, 1.0)

    @staticmethod
    def compute_coherence(text: str, llm_func: Callable[[str, str], str]) -> float:
        """
        Use LLM to check for logical contradictions.
        """
        prompt = f"""
        Analyze the following text for logical coherence:

        {text[:2000]}...

        Check for:
        1. Internal contradictions
        2. Circular reasoning
        3. Unsupported claims
        4. Logical fallacies

        Return a coherence score from 0.0 (incoherent) to 1.0 (perfectly coherent).
        Output ONLY the score as a float (e.g., 0.85).
        """

        response = llm_func("You are a logical coherence analyzer.", prompt)
        try:
            match = re.search(r"(\d\.\d+)", response)
            if match:
                return float(match.group(1))
            return 0.5  # Default if parse fails
        except Exception:
            return 0.5

    @classmethod
    def compute_epistemic_rigor(cls, text: str, llm_func: Callable[[str, str], str]) -> float:
        """
        Composite rigor metric.
        """
        specificity = cls.compute_specificity(text)
        justification = cls.compute_justification_depth(text)
        coherence = cls.compute_coherence(text, llm_func)

        # Weighted average
        rigor = 0.3 * specificity + 0.4 * justification + 0.3 * coherence

        return rigor

    @staticmethod
    def calculate_unified_score(
        c: float, r: float, context: str = "comprehensive_analysis"
    ) -> Dict[str, Any]:
        """
        Unified quality score respecting orthogonal dimensions.
        """
        # --- Robustness Checks ---
        if c is None or math.isnan(c):
            c = 0.0
        if r is None or math.isnan(r):
            r = 0.0

        # Clamp inputs to [0, 1]
        c = max(0.0, min(1.0, c))
        r = max(0.0, min(1.0, r))

        # Multiplicative interaction term (penalizes severe weakness in either dimension)
        interaction = math.sqrt(c * r)

        # Geometric mean (balanced quality)
        geometric_mean = math.sqrt(c * r)

        # Context-weighted combination
        if context == "executive_summary":
            # Breadth-first: weight coverage higher
            unified_score = 0.7 * c + 0.3 * r
        elif context == "technical_proof":
            # Depth-first: weight rigor higher
            unified_score = 0.3 * c + 0.7 * r
        elif context == "comprehensive_analysis":
            # Balanced: require both (multiplicative penalty for weakness)
            unified_score = 0.5 * (c + r) * interaction
        else:
            # Default: geometric mean (neutral balance)
            unified_score = geometric_mean

        # Quadrant classification
        if c >= 0.80 and r >= 0.80:
            quadrant = "Q2_IDEAL"
            validity = "VALID"
        elif c >= 0.60 and r >= 0.60:
            # The "Balanced/Good Enough" Zone
            quadrant = "Q2_BALANCED"
            validity = "VALID"
        elif c >= 0.60 and r < 0.60:
            quadrant = "Q1_SHALLOW"
            validity = "CONDITIONAL"
        elif c < 0.60 and r >= 0.60:
            quadrant = "Q4_DEEP"
            validity = "CONDITIONAL"
        else:
            quadrant = "Q3_INVALID"
            validity = "INVALID"

        return {
            "coverage": c,
            "rigor": r,
            "unified_score": unified_score,
            "quadrant": quadrant,
            "validity": validity,
        }

    @staticmethod
    def reconcile(c: float, r: float, context: str = "comprehensive_analysis") -> Dict[str, str]:
        """
        IF-THEN rules for resolving validator disagreements.
        """
        if c >= 0.80 and r >= 0.80:
            return {
                "decision": "APPROVE",
                "reason": "Both validators agree: high coverage and high rigor",
            }

        elif c >= 0.80 and r < 0.60:
            if context in ["executive_summary", "survey"]:
                return {
                    "decision": "APPROVE_CONDITIONAL",
                    "reason": "Shallow but appropriate for context (breadth-first)",
                    "action": "None (Context Approved)",
                }
            else:
                return {
                    "decision": "REVISE_FOR_RIGOR",
                    "reason": "High coverage but insufficient rigor for context",
                    "action": "Add: explicit premises, inference steps, concrete examples",
                }

        elif c < 0.60 and r >= 0.80:
            if context in ["technical_proof", "focused_analysis"]:
                return {
                    "decision": "APPROVE_CONDITIONAL",
                    "reason": "Deep but narrow—appropriate for specialized context",
                    "action": "None (Context Approved)",
                }
            else:
                return {
                    "decision": "REVISE_FOR_COVERAGE",
                    "reason": "High rigor but insufficient coverage for context",
                    "action": "Add: missing topics from constraint requirements",
                }

        else:  # c < 0.60 and r < 0.60
            return {
                "decision": "REJECT",
                "reason": "Both validators fail: insufficient coverage AND insufficient rigor",
                "action": "Complete rewrite required",
            }
