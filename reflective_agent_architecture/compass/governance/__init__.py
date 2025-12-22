"""
Meta-Graph Governance Module.

This package implements the Meta-System Layer (MSL) as a queryable graph of constraints.
It enforces the Triadic Kernel axioms and validates proposed actions against the Constraint DAG.
"""

import logging
from typing import Any, Dict, List

from .amendment import AmendmentController
from .ontology import OntologyManager
from .verification import ConstitutionalGuard, Violation

logger = logging.getLogger(__name__)

class MetaSystemVerifier:
    """
    Facade for the Meta-System Governance module.
    Delegates to specialized components for Ontology, Verification, and Amendments.
    """

    def __init__(self, manifold: Any, advisor_system: Any = None):
        self.logger = logging.getLogger("MetaSystemVerifier")

        # Initialize sub-components
        self.ontology = OntologyManager(manifold, advisor_system)
        self.guard = ConstitutionalGuard(manifold)
        self.amendment_controller = AmendmentController(manifold, advisor_system)

    # --- Ontology Management ---

    def bootstrap_meta_graph(self):
        """Bootstrap the Meta-Graph with the Immutable Core."""
        self.ontology.bootstrap()

    def bootstrap_source_dag(self):
        """Bootstrap the Source DAG."""
        self.ontology.bootstrap_sources()

    def generate_conceptual_graph(self, reasoning_text: str) -> Dict[str, Any]:
        """Generate a conceptual graph from reasoning text."""
        return self.ontology.generate_conceptual_graph(reasoning_text)

    # --- Verification & Intrinsic Motivation ---

    def verify_action(self, action_id: str) -> List[Violation]:
        """Verify an action against the Ontological Graph and Intrinsic Metrics."""
        return self.guard.verify_action(action_id)

    def evaluate_intrinsic_impact(self, action_id: str) -> Dict[str, float]:
        """Evaluate the intrinsic impact of an action."""
        return self.guard.evaluate_intrinsic_impact(action_id)

    def check_axioms(self) -> List[Violation]:
        """Run a global check for violations of the Triadic Kernel Axioms."""
        return self.guard.check_axioms()

    def check_epistemic_integrity(self) -> List[Violation]:
        """Ensure every fact has a source."""
        return self.guard.check_epistemic_integrity()

    def get_constraint_provenance(self, rule_text: str) -> Dict[str, Any]:
        """Retrieve the full epistemic lineage of a constraint."""
        return self.guard.get_constraint_provenance(rule_text)

    # --- Amendment Management ---

    def propose_amendment(self, amendment_text: str, justification: str, supported_axioms: List[str]) -> str:
        """Propose a constitutional amendment."""
        return self.amendment_controller.propose_amendment(amendment_text, justification, supported_axioms)

    def cleanup_expired_amendments(self):
        """Remove expired amendments."""
        self.amendment_controller.cleanup_expired_amendments()
