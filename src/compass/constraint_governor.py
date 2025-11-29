"""
Constraint Governor - Reasoning Invariants Validation

Implements the Constraint Governor module of the CGRA architecture.
Responsible for enforcing computational invariants across the reasoning stack:
1. Logical Coherence: Consistency across steps
2. Compositionality: Valid component relationships
3. Productivity: Generative capacity
4. Conceptual Processing: Abstract representation depth
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from .config import ConstraintGovernorConfig
from .utils import COMPASSLogger


@dataclass
class ConstraintViolation:
    """Record of a reasoning invariant violation."""

    invariant_type: str  # 'logical_coherence', 'compositionality', 'productivity', 'conceptual'
    description: str
    severity: str  # 'warning', 'error', 'critical'
    step_index: int
    context: Optional[Dict] = None


class ConstraintGovernor:
    """
    The Constraint Governor enforces computational invariants across the entire reasoning stack.
    It acts as a continuous validation layer.
    """

    def __init__(self, config: ConstraintGovernorConfig, logger: Optional[COMPASSLogger] = None):
        """
        Initialize the Constraint Governor.

        Args:
            config: Configuration for the governor
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or COMPASSLogger("ConstraintGovernor")

        # State tracking
        self.violations: List[ConstraintViolation] = []
        self.reasoning_history: List[Dict] = []
        self.concept_depth_history: List[float] = []

        # For logical coherence tracking
        self.active_premises: Set[str] = set()
        self.contradictions_found: int = 0

        self.logger.info("Constraint Governor initialized")

    def validate_step(self, step_data: Dict, context: Dict) -> List[ConstraintViolation]:
        """
        Validate a single reasoning step against all enabled invariants.

        Args:
            step_data: Dictionary containing step details (action, observation, reasoning)
            context: Current context dictionary

        Returns:
            List of violations found in this step
        """
        if not self.config.enable_validation:
            return []

        current_violations = []
        step_index = len(self.reasoning_history)

        # 1. Validate Logical Coherence
        if self.config.validate_logical_coherence:
            coherence_violations = self._check_logical_coherence(step_data, step_index)
            current_violations.extend(coherence_violations)

        # 2. Validate Compositionality
        if self.config.validate_compositionality:
            comp_violations = self._check_compositionality(step_data, step_index)
            current_violations.extend(comp_violations)

        # 3. Validate Productivity
        if self.config.validate_productivity:
            prod_violations = self._check_productivity(step_data, step_index)
            current_violations.extend(prod_violations)

        # 4. Validate Conceptual Processing
        if self.config.validate_conceptual_processing:
            conc_violations = self._check_conceptual_processing(step_data, step_index)
            current_violations.extend(conc_violations)

        # Update history
        self.reasoning_history.append(step_data)
        self.violations.extend(current_violations)

        if current_violations:
            self.logger.warning(f"Constraint violations in step {step_index}: {len(current_violations)}")
            for v in current_violations:
                self.logger.debug(f"  - {v.invariant_type}: {v.description}")

        return current_violations

    def _check_logical_coherence(self, step_data: Dict, step_index: int) -> List[ConstraintViolation]:
        """
        Check for logical contradictions with previous steps.

        Note: In a full implementation, this would use an NLI model or formal logic prover.
        For this MVP, we use heuristic checks and explicit contradiction markers.
        """
        violations = []
        content = str(step_data.get("reasoning", "")) + str(step_data.get("action", ""))

        # Simple heuristic: check for explicit self-contradiction phrases
        contradiction_markers = ["however, this contradicts", "but previously I said", "wait, that cannot be true because", "inconsistent with"]

        for marker in contradiction_markers:
            if marker in content.lower():
                # This might actually be GOOD meta-cognition (catching an error),
                # but it indicates a coherence break in the reasoning flow itself.
                # We flag it as a warning to ensure it's resolved.
                violations.append(ConstraintViolation(invariant_type="logical_coherence", description=f"Potential contradiction detected: '{marker}'", severity="warning", step_index=step_index))
                self.contradictions_found += 1

        # Check against max allowed contradictions
        if self.contradictions_found > self.config.max_contradictions_allowed:
            violations.append(ConstraintViolation(invariant_type="logical_coherence", description=f"Exceeded maximum allowed contradictions ({self.config.max_contradictions_allowed})", severity="error", step_index=step_index))

        return violations

    def _check_compositionality(self, step_data: Dict, step_index: int) -> List[ConstraintViolation]:
        """
        Verify that complex ideas are built from simpler components.
        Checks if new concepts introduced have defined dependencies or constituents.
        """
        violations = []

        # Check if the step has a valid structure (e.g., if it's a decomposition, does it have parts?)
        action_type = step_data.get("action_type", "")

        if action_type == "decompose":
            subtasks = step_data.get("subtasks", [])
            if not subtasks:
                violations.append(ConstraintViolation(invariant_type="compositionality", description="Decomposition action missing subtasks (compositional failure)", severity="error", step_index=step_index))

        elif action_type == "integrate":
            components = step_data.get("components", [])
            if not components:
                violations.append(ConstraintViolation(invariant_type="compositionality", description="Integration action missing components (compositional failure)", severity="error", step_index=step_index))

        return violations

    def _check_productivity(self, step_data: Dict, step_index: int) -> List[ConstraintViolation]:
        """
        Monitor generative capacity.
        Detects loops or repetitive reasoning that fails to produce new thoughts.
        """
        violations = []
        current_content = str(step_data.get("reasoning", "")).strip()

        # Check for exact repetition of previous steps (loop detection)
        # Look back at recent history
        lookback = min(step_index, 5)
        for i in range(1, lookback + 1):
            prev_step = self.reasoning_history[step_index - i]
            prev_content = str(prev_step.get("reasoning", "")).strip()

            if current_content and current_content == prev_content:
                violations.append(ConstraintViolation(invariant_type="productivity", description=f"Repetitive reasoning detected (identical to step {step_index - i})", severity="warning", step_index=step_index))
                break

        return violations

    def _check_conceptual_processing(self, step_data: Dict, step_index: int) -> List[ConstraintViolation]:
        """
        Track abstract representation depth.
        Ensures reasoning isn't just surface-level linguistic pattern matching.
        """
        violations = []

        # Heuristic: Check for conceptual depth markers
        # Deep reasoning often involves: "concept", "abstract", "model", "structure", "relationship"
        # Shallow reasoning often involves: "text", "word", "phrase", "say"

        content = str(step_data.get("reasoning", "")).lower()

        # If we have a 'conceptual_depth' metric from the model/step, use it
        if "conceptual_depth" in step_data:
            depth = step_data["conceptual_depth"]
        else:
            # Heuristic fallback based on content markers
            depth = 0.5  # Default
            deep_markers = ["concept", "abstract", "model", "structure", "relationship"]
            shallow_markers = ["text", "word", "phrase", "say"]

            if any(m in content for m in deep_markers):
                depth = 0.8
            elif any(m in content for m in shallow_markers):
                depth = 0.3

        self.concept_depth_history.append(depth)

        # Check for sustained shallow processing
        if len(self.concept_depth_history) >= 3:
            recent_avg = sum(self.concept_depth_history[-3:]) / 3
            if recent_avg < 0.2:  # Threshold for "too shallow"
                violations.append(ConstraintViolation(invariant_type="conceptual_processing", description="Sustained shallow processing detected (low conceptual depth)", severity="warning", step_index=step_index))

        return violations

    def get_violation_report(self) -> Dict:
        """Generate a summary report of all violations."""
        report = {"total_violations": len(self.violations), "by_type": {}, "by_severity": {}}

        for v in self.violations:
            # Count by type
            report["by_type"][v.invariant_type] = report["by_type"].get(v.invariant_type, 0) + 1
            # Count by severity
            report["by_severity"][v.severity] = report["by_severity"].get(v.severity, 0) + 1

        return report

    def reset(self):
        """Reset governor state."""
        self.violations.clear()
        self.reasoning_history.clear()
        self.concept_depth_history.clear()
        self.active_premises.clear()
        self.contradictions_found = 0
        self.logger.debug("Constraint Governor reset")
