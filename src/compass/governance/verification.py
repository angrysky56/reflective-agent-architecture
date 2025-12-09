"""
Verification and Intrinsic Motivation logic.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from .queries import (
    CHECK_AXIOM_1,
    CHECK_AXIOM_2,
    CHECK_AXIOM_3,
    CHECK_AXIOM_4,
    CHECK_EPISTEMIC_INTEGRITY,
    COHERENCE_CHECK,
    EMPOWERMENT_CHECK,
    GET_CONSTRAINT_PROVENANCE,
    LEGACY_CONSTRAINT_CHECK,
)

logger = logging.getLogger(__name__)


@dataclass
class Violation:
    """Represents a violation of a Meta-System constraint."""

    rule: str
    severity: str
    resolution: str
    context: Dict[str, Any]


class ConstitutionalGuard:
    """
    Verifies actions against the Meta-System Constraint DAG and Intrinsic Metrics.
    """

    def __init__(self, manifold: Any):
        self.manifold = manifold
        self.logger = logging.getLogger("ConstitutionalGuard")

    def evaluate_intrinsic_impact(self, action_id: str) -> Dict[str, float]:
        """
        Evaluates the intrinsic impact of an action on the Core Ontology.
        Returns a dictionary of metrics: coherence, empowerment.
        """
        metrics = {"coherence": 1.0, "empowerment": 0.0}

        # 1. Coherence Check: Does the action imply a Logical Contradiction?
        try:
            result = self.manifold.read_query(COHERENCE_CHECK, {"action_id": action_id})
            if result and result[0]["contradiction_count"] > 0:
                metrics["coherence"] = 0.0  # Violation!
        except Exception as e:
            self.logger.error(f"Error checking coherence: {e}")

        # 2. Empowerment Check: Does the action ENABLE concepts required by Agency?
        try:
            result = self.manifold.read_query(EMPOWERMENT_CHECK, {"action_id": action_id})
            if result:
                metrics["empowerment"] = float(result[0]["options_enabled"])
        except Exception as e:
            self.logger.error(f"Error checking empowerment: {e}")

        return metrics

    def verify_action(self, action_id: str) -> List[Violation]:
        """
        Verifies an action against the Ontological Graph and Intrinsic Metrics.
        """
        violations = []

        # 1. Intrinsic Evaluation (The "Deep Structure" Check)
        intrinsic_metrics = self.evaluate_intrinsic_impact(action_id)

        if intrinsic_metrics["coherence"] < 0.5:
            violations.append(
                Violation(
                    rule="ONTOLOGY_VIOLATION",
                    severity="CRITICAL",
                    resolution="Action implies a Logical Contradiction (Coherence Violation).",
                    context={"metrics": intrinsic_metrics},
                )
            )

        # 2. Legacy Constraint Check (The "Bureaucratic" Check - kept for specific rules)
        try:
            results = self.manifold.read_query(LEGACY_CONSTRAINT_CHECK, {"action_id": action_id})
            for record in results:
                # Format source info for transparency
                source_info = []

                for s in record["sources"]:
                    if s and s["properties"]:
                        props = s["properties"]
                        lbls = s["labels"]
                        source_info.append(
                            {
                                "title": props.get("title", "Unknown Source"),
                                "type": lbls[0] if lbls else "Source",
                                "confidence": props.get("confidence", 0.5),
                                "corroboration": props.get("corroboration_count", 0),
                            }
                        )

                violations.append(
                    Violation(
                        rule=record["rule"],
                        resolution=f"Violates constraint: {record['rule']}",
                        severity=record["severity"],
                        context={"sources": source_info},
                    )
                )
        except Exception as e:
            self.logger.error(f"Error in verify_action: {e}")
            # Fail closed if DB error
            violations.append(
                Violation(
                    "DB_ERROR",
                    "Verification failed due to database error.",
                    "CRITICAL",
                    {"error": str(e)},
                )
            )

        return violations

    def check_axioms(self) -> List[Violation]:
        """
        Run a global check for violations of the Triadic Kernel Axioms.
        """
        violations = []
        queries = [CHECK_AXIOM_1, CHECK_AXIOM_2, CHECK_AXIOM_3, CHECK_AXIOM_4]

        for q in queries:
            try:
                results = self.manifold.read_query(q)
                for record in results:
                    violations.append(
                        Violation(
                            rule=record["rule"],
                            severity="Axiomatic",
                            resolution="Refactor cognitive process to respect axiom",
                            context={"node_id": record["id"]},
                        )
                    )
            except Exception as e:
                self.logger.error(f"Axiom check error: {e}")

        return violations

    def check_epistemic_integrity(self) -> List[Violation]:
        """
        Ensure every fact has a source (Epistemic Traceability).
        """
        violations = []
        try:
            results = self.manifold.read_query(CHECK_EPISTEMIC_INTEGRITY)
            for record in results:
                violations.append(
                    Violation(
                        rule=record["rule"],
                        severity="High",
                        resolution="Attach source to fact",
                        context={"fact_id": record["id"]},
                    )
                )
        except Exception as e:
            self.logger.error(f"Epistemic check error: {e}")

        return violations

    def get_constraint_provenance(self, rule_text: str) -> Dict[str, Any]:
        """
        Retrieve the full epistemic lineage of a constraint (Transparency).
        """
        try:
            results = self.manifold.read_query(GET_CONSTRAINT_PROVENANCE, {"rule": rule_text})
            if results:
                return dict(results[0])  # Explicit cast to dict
            return {}
        except Exception as e:
            self.logger.error(f"Provenance check error: {e}")
            return {}
